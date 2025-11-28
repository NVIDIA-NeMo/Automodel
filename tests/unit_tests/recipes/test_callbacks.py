# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
from unittest.mock import MagicMock
from nemo_automodel.components.callbacks import Callback, CallbackRunner, rank_zero_only


class MockCallback(Callback):
    """Mock callback for testing that tracks which hooks were called."""
    
    def __init__(self):
        self.train_start_called = False
        self.train_batch_end_called = False
        self.validation_end_called = False
        self.save_checkpoint_called = False
        self.train_end_called = False
        self.exception_called = False
        self.received_kwargs = {}

    def on_train_start(self, recipe, **kwargs):
        self.train_start_called = True
        self.received_kwargs['train_start'] = kwargs

    def on_train_batch_end(self, recipe, **kwargs):
        self.train_batch_end_called = True
        self.received_kwargs['train_batch_end'] = kwargs

    def on_validation_end(self, recipe, **kwargs):
        self.validation_end_called = True
        self.received_kwargs['validation_end'] = kwargs

    def on_save_checkpoint(self, recipe, **kwargs):
        self.save_checkpoint_called = True
        self.received_kwargs['save_checkpoint'] = kwargs

    def on_train_end(self, recipe, **kwargs):
        self.train_end_called = True
        self.received_kwargs['train_end'] = kwargs

    def on_exception(self, recipe, **kwargs):
        self.exception_called = True
        self.received_kwargs['exception'] = kwargs


class TestCallbackSystem(unittest.TestCase):
    """Test the callback system in isolation."""
    
    def test_callback_runner_calls_all_hooks(self):
        """Test that CallbackRunner calls all callback hooks."""
        callback = MockCallback()
        runner = CallbackRunner([callback])
        recipe = MagicMock()

        runner.on_train_start(recipe)
        self.assertTrue(callback.train_start_called)

        runner.on_train_batch_end(recipe)
        self.assertTrue(callback.train_batch_end_called)

        runner.on_validation_end(recipe)
        self.assertTrue(callback.validation_end_called)

        runner.on_save_checkpoint(recipe)
        self.assertTrue(callback.save_checkpoint_called)

        runner.on_train_end(recipe)
        self.assertTrue(callback.train_end_called)

        runner.on_exception(recipe)
        self.assertTrue(callback.exception_called)
    
    def test_callback_runner_with_multiple_callbacks(self):
        """Test that multiple callbacks are all called."""
        callback1 = MockCallback()
        callback2 = MockCallback()
        runner = CallbackRunner([callback1, callback2])
        recipe = MagicMock()
        
        runner.on_train_start(recipe)
        
        self.assertTrue(callback1.train_start_called)
        self.assertTrue(callback2.train_start_called)
    
    def test_callback_runner_with_empty_list(self):
        """Test that CallbackRunner works with no callbacks."""
        runner = CallbackRunner([])
        recipe = MagicMock()
        
        # Should not raise any errors
        runner.on_train_start(recipe)
        runner.on_train_batch_end(recipe)
        runner.on_validation_end(recipe)
        runner.on_save_checkpoint(recipe)
        runner.on_exception(recipe)
    
    def test_callback_receives_kwargs(self):
        """Test that callbacks receive hook-specific kwargs."""
        callback = MockCallback()
        runner = CallbackRunner([callback])
        recipe = MagicMock()
        
        # Test train_batch_end with train_log_data
        mock_log_data = MagicMock()
        runner.on_train_batch_end(recipe, train_log_data=mock_log_data)
        self.assertIn('train_log_data', callback.received_kwargs['train_batch_end'])
        self.assertEqual(callback.received_kwargs['train_batch_end']['train_log_data'], mock_log_data)
        
        # Test validation_end with val_results
        val_results = {'validation': MagicMock()}
        runner.on_validation_end(recipe, val_results=val_results)
        self.assertIn('val_results', callback.received_kwargs['validation_end'])
        self.assertEqual(callback.received_kwargs['validation_end']['val_results'], val_results)
        
        # Test exception with exception object
        exception = RuntimeError("Test error")
        runner.on_exception(recipe, exception=exception)
        self.assertIn('exception', callback.received_kwargs['exception'])
        self.assertEqual(callback.received_kwargs['exception']['exception'], exception)
        
        # Test save_checkpoint with checkpoint_info
        checkpoint_info = MagicMock()
        runner.on_save_checkpoint(recipe, checkpoint_info=checkpoint_info)
        self.assertIn('checkpoint_info', callback.received_kwargs['save_checkpoint'])
        self.assertEqual(callback.received_kwargs['save_checkpoint']['checkpoint_info'], checkpoint_info)
    
    def test_callback_execution_order(self):
        """Test that callbacks are executed in the order they're provided."""
        call_order = []
        
        class OrderTracker1(Callback):
            def on_train_start(self, recipe, **kwargs):
                call_order.append(1)
        
        class OrderTracker2(Callback):
            def on_train_start(self, recipe, **kwargs):
                call_order.append(2)
        
        runner = CallbackRunner([OrderTracker1(), OrderTracker2()])
        runner.on_train_start(MagicMock())
        
        self.assertEqual(call_order, [1, 2])
    
    def test_callback_base_class_does_nothing(self):
        """Test that the base Callback class does nothing by default."""
        callback = Callback()
        recipe = MagicMock()
        
        # Should not raise any errors
        callback.on_train_start(recipe)
        callback.on_train_batch_end(recipe)
        callback.on_validation_end(recipe)
        callback.on_save_checkpoint(recipe)
        callback.on_train_end(recipe)
        callback.on_exception(recipe)
    
    def test_rank_zero_only_decorator_on_rank_zero(self):
        """Test that @rank_zero_only allows execution on rank 0."""
        class RankZeroCallback(Callback):
            def __init__(self):
                self.called = False
            
            @rank_zero_only
            def on_train_start(self, recipe, **kwargs):
                self.called = True
        
        callback = RankZeroCallback()
        
        # Mock recipe with rank 0 (main process)
        recipe = MagicMock()
        recipe.dist_env.is_main = True
        
        callback.on_train_start(recipe)
        self.assertTrue(callback.called)
    
    def test_rank_zero_only_decorator_on_non_zero_rank(self):
        """Test that @rank_zero_only blocks execution on non-zero ranks."""
        class RankZeroCallback(Callback):
            def __init__(self):
                self.called = False
            
            @rank_zero_only
            def on_train_start(self, recipe, **kwargs):
                self.called = True
        
        callback = RankZeroCallback()
        
        # Mock recipe with rank 1 (not main process)
        recipe = MagicMock()
        recipe.dist_env.is_main = False
        
        callback.on_train_start(recipe)
        self.assertFalse(callback.called)
    
    def test_rank_zero_only_decorator_without_dist_env(self):
        """Test that @rank_zero_only works when dist_env is missing (single GPU)."""
        class RankZeroCallback(Callback):
            def __init__(self):
                self.called = False
            
            @rank_zero_only
            def on_train_start(self, recipe, **kwargs):
                self.called = True
        
        callback = RankZeroCallback()
        
        # Mock recipe without dist_env (e.g., single GPU training)
        recipe = MagicMock(spec=[])  # Empty spec means no attributes
        
        callback.on_train_start(recipe)
        # Should still execute when dist_env is not present
        self.assertTrue(callback.called)


if __name__ == '__main__':
    unittest.main()
