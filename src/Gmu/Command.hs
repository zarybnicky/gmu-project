{-# LANGUAGE OverloadedLists #-}

module Gmu.Command
  ( oneshot
  ) where

import Control.Monad.Reader.Class (asks)
import Data.Word (Word64)
import Gmu.Monad
import Vulkan.Core10
import Vulkan.CStruct.Extends
import Vulkan.Zero
import Control.Exception.Safe (bracket, throwString)

oneshot :: Word64 -> (CommandBuffer -> Vulkan ()) -> Vulkan ()
oneshot timeout action = do
  device <- asks ghDevice
  queue <- asks ghQueue
  commandPool <- asks ghCommandPool
  let commandBufferAllocateInfo = zero
        { commandPool = commandPool
        , level = COMMAND_BUFFER_LEVEL_PRIMARY
        , commandBufferCount = 1
        }
  withCommandBuffers device commandBufferAllocateInfo bracket $ \case
    [buf] -> do
      useCommandBuffer buf oneTime $ action buf
      withFence device zero Nothing bracket $ \fence -> do
        queueSubmit queue [SomeStruct zero { commandBuffers = [commandBufferHandle buf] }] fence
        waitForFences device [fence] True timeout >>= \case
          SUCCESS -> pure ()
          TIMEOUT -> throwString "Timed out waiting for compute"
          err -> throwString ("Error in FENCE_CREATE_SIGNALED_BIT failed: " <> show err)
    _ -> error "assert: exactly the requested buffer was given"
  pure ()
  where
    oneTime :: CommandBufferBeginInfo '[]
    oneTime = zero { flags = COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT }
