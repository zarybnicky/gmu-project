{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE OverloadedStrings #-}

module Gmu.Instance
  ( PhysicalDeviceInfo(..)
  , createInstanceApp
  , createDeviceApp
  ) where

import Control.Exception.Safe (throwString)
import Control.Monad.IO.Class (MonadIO)
import Control.Monad.Trans.Maybe (MaybeT(..))
import Control.Monad.Trans.Resource (MonadResource, MonadThrow, allocate)
import Data.Bits ((.|.), (.&.), zeroBits)
import Data.Foldable (maximumBy)
import Data.Maybe (catMaybes)
import Data.Ord (comparing)
import Data.Text (Text)
import Data.Text.Encoding (decodeUtf8)
import qualified Data.Vector as V
import Data.Word
import Say
import Vulkan.CStruct.Extends
import Vulkan.Core10 as Vk
import Vulkan.Extensions.VK_EXT_debug_utils
import Vulkan.Zero (Zero(zero))
import Vulkan.Utils.Debug (debugCallbackPtr)

createInstanceApp :: MonadResource m => m Instance
createInstanceApp = do
  let debugMessengerCreateInfo = zero
        { messageSeverity = DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
                        .|. DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT
        , messageType     = DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
                        .|. DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
                        .|. DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT
        , pfnUserCallback = debugCallbackPtr
        }
      instanceCreateInfo =
        (zero { enabledLayerNames     = []
              , enabledExtensionNames = [EXT_DEBUG_UTILS_EXTENSION_NAME]
              } :: InstanceCreateInfo '[])
          ::& debugMessengerCreateInfo :& ()
  (_, inst) <- withInstance instanceCreateInfo Nothing allocate
  _ <- withDebugUtilsMessengerEXT inst debugMessengerCreateInfo Nothing allocate
  pure inst

createDeviceApp :: (MonadResource m, MonadThrow m) => Instance -> m (PhysicalDevice, PhysicalDeviceInfo, Device)
createDeviceApp inst = do
  (pdi, phys) <- pickPhysicalDevice inst physicalDeviceInfo
  sayErr . ("Using device: " <>) =<< physicalDeviceName phys
  let computeInfo = zero
        { queueFamilyIndex = pdiComputeQueueFamilyIndex pdi
        , queuePriorities  = [1]
        }
  let deviceCreateInfo = zero { queueCreateInfos = [SomeStruct computeInfo] }
  (_, dev) <- withDevice phys deviceCreateInfo Nothing allocate
  pure (phys, pdi, dev)

pickPhysicalDevice
  :: (MonadIO m, MonadThrow m, Ord a)
  => Instance
  -> (PhysicalDevice -> m (Maybe a))
  -> m (a, PhysicalDevice)
pickPhysicalDevice inst devScore = do
  (_, devs) <- enumeratePhysicalDevices inst
  scores <- catMaybes <$> sequence [fmap (, d) <$> devScore d | d <- V.toList devs]
  case scores of
    [] -> throwString "Unable to find appropriate PhysicalDevice"
    _ -> pure (maximumBy (comparing fst) scores)

data PhysicalDeviceInfo = PhysicalDeviceInfo
  { pdiTotalMemory :: Word64
  , pdiComputeQueueFamilyIndex :: Word32
  } deriving (Eq, Ord)

physicalDeviceInfo :: MonadIO m => PhysicalDevice -> m (Maybe PhysicalDeviceInfo)
physicalDeviceInfo phys = runMaybeT $ do
  pdiTotalMemory <- do
    heaps <- memoryHeaps <$> getPhysicalDeviceMemoryProperties phys
    pure $ sum ((size :: MemoryHeap -> DeviceSize) <$> heaps)
  pdiComputeQueueFamilyIndex <- do
    queueFamilyProperties <- getPhysicalDeviceQueueFamilyProperties phys
    let isComputeQueue q = (zeroBits /= QUEUE_COMPUTE_BIT .&. queueFlags q) && (queueCount q > 0)
        computeQueueIndices = fromIntegral . fst <$> V.filter
          (isComputeQueue . snd)
          (V.indexed queueFamilyProperties)
    MaybeT (pure $ computeQueueIndices V.!? 0)
  pure PhysicalDeviceInfo { .. }

physicalDeviceName :: MonadIO m => PhysicalDevice -> m Text
physicalDeviceName = fmap (decodeUtf8 . deviceName) . getPhysicalDeviceProperties
