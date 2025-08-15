---
description: "Learn about our platform's core concepts, key features, and fundamental architecture to understand how it works."
tags: ["overview", "concepts", "architecture", "features"]
categories: ["concepts"]
---

(about-overview)=
# About NeMo NeMo Automodel

NVIDIA NeMo NeMo Automodel bridges Hugging Face models with NVIDIA's high-performance training ecosystem, providing Day-0 support for new models without conversions or rewrites. It offers a user-friendly interface while delivering enterprise-grade scalability and performance optimizations.

## What is NeMo Automodel?

NeMo Automodel provides immediate access to Hugging Face models within the NeMo framework, combining HF's breadth with NeMo's performance and scaling capabilities. Key capabilities include accelerated training with BF16/FP8, distributed scaling via DDP and FSDP2, PEFT/LoRA for efficient fine-tuning, and YAML-driven recipes for reproducible workflows.

## How it Works

1. **Load a Hugging Face model** using NeMo NeMo Automodel's drop-in interface
2. **Prepare your dataset** with Hugging Face datasets and NeMo utilities  
3. **Configure training** via YAML with model, data, parallelism, and fine-tuning settings
4. **Train and deploy** leveraging NeMo's optimizations and export to inference frameworks

## Scope and Support

NeMo Automodel currently supports:
- **Large Language Models (LLMs)**: Text generation and instruction following
- **Vision Language Models (VLMs)**: Image-text understanding and generation  
- **Roadmap**: Video generation and additional modalities

## Why Day-0 Access Matters

The rapid pace of open-source model releases—including Meta Llama, Google Gemma, Mistral models, Qwen series, DeepSeek R1, and NVIDIA Nemotron—creates opportunities for immediate experimentation and fine-tuning. NeMo Automodel eliminates the traditional delay between model release and optimized training recipes, enabling teams to harness innovations efficiently while maintaining a path to Megatron-Core optimizations as they become available.

## Target Users

NeMo Automodel is designed for various AI practitioners who need efficient, scalable training solutions:

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} {octicon}`person;1.5em;sd-mr-1` Researchers
:class-header: sd-bg-primary sd-text-white

Experiment with cutting-edge models immediately upon release for research projects and paper development.
+++
{bdg-primary}`Research`
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` ML Engineers
:class-header: sd-bg-success sd-text-white

Build production-ready training pipelines with enterprise-grade scalability and performance optimization.
+++
{bdg-success}`Production`
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Data Scientists
:class-header: sd-bg-info sd-text-white

Fine-tune models for specific tasks using PEFT techniques and ready-to-use configurations.
+++
{bdg-info}`Fine-tuning`
:::

:::{grid-item-card} {octicon}`cloud;1.5em;sd-mr-1` DevOps Teams
:class-header: sd-bg-warning sd-text-white

Deploy scalable training infrastructure with containerized environments and cluster management.
+++
{bdg-warning}`Operations`
:::


## Explore NeMo NeMo Automodel

::::{grid} 1 1 2 3
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`star;1.5em;sd-mr-1` Key Features
:link: key-features
:link-type: doc
:link-alt: Detailed feature comparison and capabilities

Comprehensive overview of capabilities, performance optimizations, and backend comparisons.
+++
{bdg-info}`Features`
:::

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Architecture Overview
:link: architecture-overview
:link-type: doc
:link-alt: Technical architecture details

Deep dive into the modular architecture, components, and design principles.
+++
{bdg-secondary}`Technical`
:::

:::{grid-item-card} {octicon}`repo;1.5em;sd-mr-1` Repository & Package Guide
:link: repository-and-package-guide
:link-type: doc
:link-alt: Complete codebase organization guide

Comprehensive guide to repository structure, package hierarchy, and development patterns.
+++
{bdg-primary}`Structure`
:::

:::{grid-item-card} {octicon}`play;1.5em;sd-mr-1` Get Started
:link: /get-started/index
:link-type: doc
:link-alt: Installation and quick start guide

Installation guide and hands-on tutorial to begin fine-tuning models.
+++
{bdg-success}`Tutorial`
:::

::::

## Package Components

Explore the core components that make NeMo Automodel powerful and flexible:

::::{grid} 1 1 2 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Core Components
:link: /api-docs/components/components
:link-type: doc
:link-alt: Core training components

Modular components including transformers, PEFT, datasets, distributed training, loss functions, and training utilities.

+++
{bdg-primary}`Components`
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Training Recipes
:link: /api-docs/recipes/recipes
:link-type: doc
:link-alt: End-to-end training workflows

Complete training pipelines for LLM and VLM fine-tuning with configurable workflows and best practices.

+++
{bdg-info}`Recipes`
:::

:::{grid-item-card} {octicon}`terminal;1.5em;sd-mr-1` Command Line Interface
:link: /api-docs/cli/index
:link-type: doc
:link-alt: CLI tools and launcher

Powerful CLI for launching distributed training jobs across different environments and cluster systems.

+++
{bdg-warning}`CLI`
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Shared Utilities
:link: /api-docs/shared/shared
:link-type: doc
:link-alt: Common utilities and helpers

Import management, configuration utilities, and shared functionality used across the package.

+++
{bdg-secondary}`Utils`
:::

::::

### Key Functionality Areas

#### 🤖 **Model Integration**
- **Transformers**: Drop-in replacements for Hugging Face models (`NeMoAutoModelForCausalLM`, `NeMoAutoModelForImageTextToText`)
- **PEFT Support**: LoRA implementation with optimized kernels for parameter-efficient fine-tuning
- **Day-0 Compatibility**: Immediate support for new Hugging Face model releases

#### 📊 **Data & Datasets**  
- **LLM Datasets**: Instruction datasets, HellaSwag, SQuAD, packed sequences
- **VLM Datasets**: Vision-language datasets with specialized collation functions
- **Processing Utilities**: Data preprocessing and transformation tools

#### ⚡ **Distributed Training**
- **Parallelism**: DDP, FSDP2, nvFSDP, and tensor parallelism strategies
- **Optimization**: Gradient utilities, distributed communication, and scaling
- **Cluster Support**: SLURM integration and multi-node training capabilities

#### 🔧 **Training Infrastructure**
- **Advanced Checkpointing**: HuggingFace-compatible checkpointing with state management
- **Loss Functions**: Optimized cross-entropy variants with Transformer Engine support
- **Quantization**: FP8 quantization for memory efficiency and speed
- **Monitoring**: WandB integration and comprehensive logging

### Component Architecture

See how NeMo Automodel components work together:

```{raw} html
<div id="component-architecture-modal-container" style="cursor: pointer; border: 2px solid transparent; border-radius: 8px; transition: border-color 0.3s ease; padding: 10px;" 
     onmouseover="this.style.borderColor='#007acc'; this.style.backgroundColor='rgba(0,122,204,0.05)'" 
     onmouseout="this.style.borderColor='transparent'; this.style.backgroundColor='transparent'"
     onclick="openDiagramModal('component-architecture-modal', 'Component Architecture Diagram')">
```

```{mermaid}
graph TD
    A["🤖 Model Classes<br/>NeMoAutoModelForCausalLM<br/>NeMoAutoModelForImageTextToText"] --> B["📊 Datasets<br/>LLM & VLM<br/>Data Processing"]
    A --> C["⚡ Distributed Training<br/>DDP, FSDP2, nvFSDP<br/>Tensor Parallelism"]
    A --> D["🔧 PEFT<br/>LoRA Implementation<br/>Efficient Fine-tuning"]
    
    B --> E["🍳 Training Recipes<br/>LLM Finetune<br/>VLM Finetune"]
    C --> E
    D --> E
    
    E --> F["💾 Checkpointing<br/>State Management<br/>HF Integration"]
    E --> G["📈 Loss Functions<br/>Optimized CE<br/>TE Parallel"]
    E --> H["📊 Logging<br/>WandB Integration<br/>Monitoring"]
    
    I["🔧 Shared Utilities<br/>Import Utils<br/>Config Management"] --> A
    I --> B
    I --> C
    I --> D
    
    J["💻 CLI Interface<br/>Job Launcher<br/>SLURM Integration"] --> E
    
    style A fill:#e1f5fe
    style E fill:#f3e5f5
    style I fill:#f1f8e9
    style J fill:#fff3e0
```

```{raw} html
</div>

<!-- Modal for Component Architecture Diagram -->
<div id="component-architecture-modal" style="display: none; position: fixed; z-index: 10000; left: 0; top: 0; width: 100%; height: 100%; background-color: rgba(0,0,0,0.8); backdrop-filter: blur(3px);" onclick="closeDiagramModal('component-architecture-modal')">
    <div style="position: relative; margin: auto; padding: 20px; width: 95%; max-width: 1400px; top: 50%; transform: translateY(-50%);">
        <span onclick="closeDiagramModal('component-architecture-modal')" style="color: white; float: right; font-size: 28px; font-weight: bold; cursor: pointer; background: rgba(0,0,0,0.5); padding: 5px 10px; border-radius: 50%;">&times;</span>
        <div style="text-align: center; max-height: 80vh; overflow-y: auto; background: white; border-radius: 12px; padding: 20px;">
            <h3 id="component-architecture-modal-title" style="margin-top: 0; color: #333;"></h3>
            <div id="component-architecture-modal-content" style="transform: scale(1.5); transform-origin: center; margin: 40px 0;">
```

```{mermaid}
graph TD
    A["🤖 Model Classes<br/>NeMoAutoModelForCausalLM<br/>NeMoAutoModelForImageTextToText"] --> B["📊 Datasets<br/>LLM & VLM<br/>Data Processing"]
    A --> C["⚡ Distributed Training<br/>DDP, FSDP2, nvFSDP<br/>Tensor Parallelism"]
    A --> D["🔧 PEFT<br/>LoRA Implementation<br/>Efficient Fine-tuning"]
    
    B --> E["🍳 Training Recipes<br/>LLM Finetune<br/>VLM Finetune"]
    C --> E
    D --> E
    
    E --> F["💾 Checkpointing<br/>State Management<br/>HF Integration"]
    E --> G["📈 Loss Functions<br/>Optimized CE<br/>TE Parallel"]
    E --> H["📊 Logging<br/>WandB Integration<br/>Monitoring"]
    
    I["🔧 Shared Utilities<br/>Import Utils<br/>Config Management"] --> A
    I --> B
    I --> C
    I --> D
    
    J["💻 CLI Interface<br/>Job Launcher<br/>SLURM Integration"] --> E
    
    style A fill:#e1f5fe
    style E fill:#f3e5f5
    style I fill:#f1f8e9
    style J fill:#fff3e0
```

```{raw} html
            </div>
        </div>
    </div>
</div>

<script>
function openDiagramModal(modalId, title) {
    const modal = document.getElementById(modalId);
    const modalTitle = document.getElementById(modalId + '-title');
    
    modal.style.display = 'block';
    modalTitle.textContent = title;
    
    // Prevent body scroll when modal is open
    document.body.style.overflow = 'hidden';
}

function closeDiagramModal(modalId) {
    const modal = document.getElementById(modalId);
    modal.style.display = 'none';
    document.body.style.overflow = 'auto';
}

// Close modal with Escape key (unified with image modals)
document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape') {
        const modals = document.querySelectorAll('[id*="-modal"]');
        modals.forEach(modal => {
            if (modal.style.display === 'block') {
                modal.style.display = 'none';
                document.body.style.overflow = 'auto';
            }
        });
    }
});
</script>
```

## Quick Links

- **Day-0 Support**: Use new HF models immediately without conversions
- **Performance**: BF16/FP8 quantization, optimized attention, Liger kernels
- **Scaling**: DDP, FSDP2, nvFSDP for single GPU to multi-node clusters
- **Fine-tuning**: Both full fine-tuning (SFT) and parameter-efficient (PEFT/LoRA)
- **Export**: Deploy to vLLM, TensorRT-LLM, and other inference frameworks

## Get Started

Ready to begin training with NeMo Automodel? Follow these essential resources to get up and running:

### Essential Resources

::::{grid} 1 1 2 3
:gutter: 2

:::{grid-item-card} {octicon}`download;1.5em;sd-mr-1` Installation
:link: /get-started/installation
:link-type: doc
:link-alt: Complete installation guide

Set up NeMo Automodel with container, pip, or development installation options.
+++
{bdg-primary}`Step 1`
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Quick Start
:link: /get-started/index
:link-type: doc
:link-alt: Hands-on tutorial

Follow the quick start tutorial to fine-tune your first model in minutes.
+++
{bdg-success}`Step 2`
:::

:::{grid-item-card} {octicon}`code;1.5em;sd-mr-1` Examples
:link: /learning-resources/examples/index
:link-type: doc
:link-alt: Working examples

Explore ready-to-run examples for LLM and VLM fine-tuning workflows.
+++
{bdg-info}`Step 3`
:::

::::

### Model-Specific Guides

- **[LLM Training](/guides/llm/index)** - Language model fine-tuning with SFT and PEFT
- **[VLM Training](/guides/vlm/index)** - Vision-language model training workflows  
- **[Model Coverage](/model-coverage/llm)** - Comprehensive list of supported model architectures
- **[Distributed Training](/guides/launcher/index)** - Multi-GPU and cluster deployment

### API Documentation

- **[Components API](/api-docs/components/index)** - Modular training components reference
- **[Recipes API](/api-docs/recipes/index)** - End-to-end workflow documentation
- **[CLI Reference](/api-docs/cli/index)** - Command-line interface guide
