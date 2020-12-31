{-# LANGUAGE TemplateHaskell #-}

module Gmu.Monad
  ( GlobalHandles(..)
  , Vulkan(..)
  , runVulkan
  ) where

import Control.Exception.Safe (finally, MonadCatch, MonadMask)
import Control.Monad.IO.Class (MonadIO)
import Control.Monad.Reader.Class (MonadReader)
import Control.Monad.Trans.Reader (ReaderT(..))
import Control.Monad.Trans.Resource (allocate, runResourceT, MonadResource, MonadThrow, ResourceT)
import Data.Word (Word32)

import Vulkan.Core10 as Vk hiding (withBuffer, withImage)
import Vulkan.Zero (Zero(zero))
import VulkanMemoryAllocator as VMA hiding (getPhysicalDeviceProperties)

import Gmu.Instance (pdiComputeQueueFamilyIndex, PhysicalDeviceInfo)

newtype Vulkan a = Vulkan { unVulkan :: ReaderT GlobalHandles (ResourceT IO) a }
  deriving (Functor, Applicative, Monad, MonadFail, MonadThrow,
            MonadCatch, MonadMask, MonadIO, MonadResource,
            MonadReader GlobalHandles)

runVulkan
  :: ResourceT IO Instance
  -> (Instance -> ResourceT IO (PhysicalDevice, PhysicalDeviceInfo, Device))
  -> Vulkan a
  -> IO a
runVulkan instanceM deviceM f = runResourceT $ do
  ghInstance <- instanceM
  (ghPhysicalDevice, pdi, ghDevice) <- deviceM ghInstance
  (_, ghAllocator) <- withAllocator
    zero { flags = zero
         , physicalDevice = physicalDeviceHandle ghPhysicalDevice
         , device = deviceHandle ghDevice
         , instance' = instanceHandle ghInstance
         }
    allocate
  let ghComputeQueueFamilyIndex = pdiComputeQueueFamilyIndex pdi
  ghQueue <- getDeviceQueue ghDevice ghComputeQueueFamilyIndex 0
  (_, ghCommandPool) <- withCommandPool
    ghDevice
    zero { queueFamilyIndex = ghComputeQueueFamilyIndex }
    Nothing
    allocate
  runReaderT (unVulkan f `finally` deviceWaitIdle ghDevice) GlobalHandles {..}

data GlobalHandles = GlobalHandles
  { ghInstance :: Instance
  , ghPhysicalDevice :: PhysicalDevice
  , ghDevice :: Device
  , ghAllocator :: Allocator
  , ghComputeQueueFamilyIndex :: Word32
  , ghCommandPool :: CommandPool
  , ghQueue :: Queue
  }
