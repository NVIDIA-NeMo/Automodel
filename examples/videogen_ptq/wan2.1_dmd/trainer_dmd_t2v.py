# trainer_dmd_t2v.py - Key changes for pure DMD

# In __init__, remove denoising_step_list and self-forcing params:
def __init__(
    self,
    model_id: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    teacher_model_path: Optional[str] = None,
    learning_rate: float = 1e-5,
    critic_learning_rate: Optional[float] = None,
    cpu_offload: bool = True,
    # DMD parameters
    num_train_timestep: int = 1000,
    min_step: int = 20,
    max_step: int = 980,
    real_guidance_scale: float = 5.0,
    fake_guidance_scale: float = 0.0,
    timestep_shift: float = 3.0,
    loss_weight_type: str = "constant",  # NEW: "constant" or "sigma"
    loss_weight_scale: float = 1.0,      # NEW: weight scaling factor
    # Alternating optimization
    critic_steps: int = 2,  # Train critic more initially
    student_steps: int = 1,
):
    # ... rest of init ...
    
    # Remove denoising_step_list
    # self.denoising_step_list = ...  # DELETE THIS
    
    # Add new params
    self.loss_weight_type = loss_weight_type
    self.loss_weight_scale = loss_weight_scale
    self.critic_steps = critic_steps
    self.student_steps = student_steps

# In setup_dmd_model, pass new params:
def setup_dmd_model(self):
    """Initialize DMD model (pure DMD, no self-forcing)."""
    print0("[INFO] Setting up pure DMD model...")
    
    self.dmd_model = DMDT2V(
        model_map=self.model_map,
        scheduler=self.pipe.scheduler,
        device=self.device,
        bf16=self.bf16,
        num_train_timestep=self.num_train_timestep,
        min_step=self.min_step,
        max_step=self.max_step,
        real_guidance_scale=self.real_guidance_scale,
        fake_guidance_scale=self.fake_guidance_scale,
        timestep_shift=self.timestep_shift,
        loss_weight_type=self.loss_weight_type,
        loss_weight_scale=self.loss_weight_scale,
    )
    
    print0("[INFO] Pure DMD model initialized")

# Add setup_pipeline method:
def setup_pipeline(self):
    """Setup simple DMD pipeline."""
    from pipeline_t2v import SimpleDMDPipeline
    
    print0("[INFO] Setting up simple DMD pipeline...")
    
    self.pipeline = SimpleDMDPipeline(
        scheduler=self.pipe.scheduler,
        student_model=self.model_map["generator"]["fsdp_transformer"],
    )
    
    print0("[INFO] Pipeline initialized")

# In train method, update the training loop:
def train(self, ...):
    # ... setup code ...
    
    # Setup components
    self.setup_pipeline()  # Load base pipeline first
    self.setup_fsdp()
    self.setup_dmd_model()
    self.setup_pipeline()  # Create SimpleDMDPipeline after FSDP
    self.setup_optim()
    self.validate_setup()
    
    # ... dataloader and scheduler setup ...
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        iterable = dataloader
        if is_main_process():
            from tqdm import tqdm
            iterable = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        
        epoch_student_loss = 0.0
        epoch_critic_loss = 0.0
        num_steps = 0
        
        for step, batch in enumerate(iterable):
            # Alternating optimization
            # Update critic for critic_steps
            for _ in range(self.critic_steps):
                self.critic_optimizer.zero_grad(set_to_none=True)
                
                try:
                    _, critic_loss, crit_metrics = step_dmd_alternating(
                        dmd_model=self.dmd_model,
                        pipeline=self.pipeline,
                        batch=batch,
                        device=self.device,
                        bf16=self.bf16,
                        global_step=global_step,
                        update_student=False,
                        update_critic=True,
                    )
                    
                    if critic_loss is not None:
                        critic_loss.backward()
                        
                        # Gradient clipping
                        critic_params = [
                            p for p in self.model_map["fake_score"]["fsdp_transformer"].parameters()
                            if p.requires_grad and p.grad is not None
                        ]
                        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                            critic_params, max_norm=1.0
                        )
                        
                        self.critic_optimizer.step()
                        epoch_critic_loss += critic_loss.item()
                
                except Exception as e:
                    print0(f"[ERROR] Critic step failed: {e}")
                    raise
            
            # Update student for student_steps
            for _ in range(self.student_steps):
                self.student_optimizer.zero_grad(set_to_none=True)
                
                try:
                    student_loss, _, stud_metrics = step_dmd_alternating(
                        dmd_model=self.dmd_model,
                        pipeline=self.pipeline,
                        batch=batch,
                        device=self.device,
                        bf16=self.bf16,
                        global_step=global_step,
                        update_student=True,
                        update_critic=False,
                    )
                    
                    if student_loss is not None:
                        student_loss.backward()
                        
                        # Gradient clipping
                        student_params = [
                            p for p in self.model_map["generator"]["fsdp_transformer"].parameters()
                            if p.requires_grad and p.grad is not None
                        ]
                        student_grad_norm = torch.nn.utils.clip_grad_norm_(
                            student_params, max_norm=1.0
                        )
                        
                        self.student_optimizer.step()
                        self.lr_scheduler.step()
                        epoch_student_loss += student_loss.item()
                
                except Exception as e:
                    print0(f"[ERROR] Student step failed: {e}")
                    raise
            
            num_steps += 1
            global_step += 1
            
            # Logging
            if is_main_process() and (global_step % log_every == 0):
                log_dict = {
                    "student_loss": student_loss.item() if student_loss is not None else 0.0,
                    "critic_loss": critic_loss.item() if critic_loss is not None else 0.0,
                    "avg_student_loss": epoch_student_loss / (num_steps * self.student_steps),
                    "avg_critic_loss": epoch_critic_loss / (num_steps * self.critic_steps),
                    "student_lr": self.student_optimizer.param_groups[0]["lr"],
                    "critic_lr": self.critic_optimizer.param_groups[0]["lr"],
                    "student_grad_norm": float(student_grad_norm) if "student_grad_norm" in locals() else 0.0,
                    "critic_grad_norm": float(critic_grad_norm) if "critic_grad_norm" in locals() else 0.0,
                    "epoch": epoch,
                    "global_step": global_step,
                }
                
                wandb.log(log_dict, step=global_step)
                
                if hasattr(iterable, "set_postfix"):
                    iterable.set_postfix({
                        "s_loss": f"{student_loss.item() if student_loss else 0:.4f}",
                        "c_loss": f"{critic_loss.item() if critic_loss else 0:.4f}",
                        "lr": f"{log_dict['student_lr']:.2e}",
                    })
            
            # Checkpointing
            if save_every and (global_step % save_every == 0):
                save_fsdp_dmd_checkpoint(
                    self.model_map,
                    self.student_optimizer,
                    self.critic_optimizer,
                    self.lr_scheduler,
                    output_dir,
                    global_step,
                    consolidate=True,
                )
        
        # Epoch summary
        avg_student_loss = epoch_student_loss / max(num_steps * self.student_steps, 1)
        avg_critic_loss = epoch_critic_loss / max(num_steps * self.critic_steps, 1)
        print0(f"[INFO] Epoch {epoch + 1} complete")
        print0(f"  Student loss: {avg_student_loss:.6f}")
        print0(f"  Critic loss: {avg_critic_loss:.6f}")
        
        if is_main_process():
            wandb.log({
                "epoch/avg_student_loss": avg_student_loss,
                "epoch/avg_critic_loss": avg_critic_loss,
                "epoch/num": epoch + 1,
            }, step=global_step)
    
    # Final checkpoint
    if is_main_process():
        print0("[INFO] Training complete, saving final checkpoint...")
        save_fsdp_dmd_checkpoint(
            self.model_map,
            self.student_optimizer,
            self.critic_optimizer,
            self.lr_scheduler,
            output_dir,
            global_step,
            consolidate=True,
        )
        wandb.finish()
    
    print0("[INFO] Pure DMD training complete!")