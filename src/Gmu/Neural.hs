{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE TemplateHaskell #-}
{-# OPTIONS_GHC -Wno-missing-signatures #-}

module Gmu.Neural
  ( runNn
  ) where

import qualified Codec.Picture as JP
import Control.Monad.IO.Class
import Control.Monad.Reader.Class (asks)
import Control.Monad.Trans.Resource
import qualified Data.ByteString.Lazy as BSL
import Foreign.Marshal.Array (peekArray)
import Foreign.Ptr
import Foreign.Storable (sizeOf)
import Say

import Vulkan.CStruct.Extends
import Vulkan.CStruct.Utils (FixedArray, lowerArrayPtr)
import Vulkan.Core10 as Vk hiding (withBuffer, withImage)
import Vulkan.Zero
import VulkanMemoryAllocator as VMA hiding (getPhysicalDeviceProperties)

import Gmu.Command (oneshot)
import Gmu.Instance (createInstanceApp, createDeviceApp)
import Gmu.Monad (GlobalHandles(..), Vulkan, runVulkan)
import Gmu.Quasi (compileShaderQ, glslFile)


runNn :: IO ()
runNn = runVulkan createInstanceApp createDeviceApp $ do
  image <- render
  let filename = "julia.png"
  sayErr $ "Writing " <> filename
  liftIO $ BSL.writeFile filename (JP.encodePng image)

render :: Vulkan (JP.Image JP.PixelRGBA8)
render = do
  let width, height, workgroupX, workgroupY :: Int
      width = 512
      height = width
      workgroupX = 32
      workgroupY = 4

  -- Create a buffer into which to render
  allocator <- asks ghAllocator
  (_, (buffer, bufferAllocation, bufferAllocationInfo)) <- withBuffer allocator
    zero { size  = fromIntegral $ width * height * 4 * sizeOf (0 :: Float)
         , usage = BUFFER_USAGE_STORAGE_BUFFER_BIT
         }
    zero { flags = ALLOCATION_CREATE_MAPPED_BIT
         , usage = MEMORY_USAGE_GPU_TO_CPU
         }
    allocate

  -- Create a descriptor set and layout for this buffer
  device <- asks ghDevice
  (descriptorSet, descriptorSetLayout) <- do
    let poolDesc = zero
          { maxSets = 1
          , poolSizes = [DescriptorPoolSize DESCRIPTOR_TYPE_STORAGE_BUFFER 1]
          }
    (_, descriptorPool) <- withDescriptorPool device poolDesc Nothing allocate

    let computeStorage = zero
          { binding = 0
          , descriptorType = DESCRIPTOR_TYPE_STORAGE_BUFFER
          , descriptorCount = 1
          , stageFlags = SHADER_STAGE_COMPUTE_BIT
          }
    (_, descriptorSetLayout) <- withDescriptorSetLayout
      device zero { bindings = [computeStorage] } Nothing allocate

    -- Allocate a descriptor set from the pool with that layout
    [descriptorSet] <- allocateDescriptorSets device zero
      { descriptorPool = descriptorPool
      , setLayouts = [descriptorSetLayout]
      }
    pure (descriptorSet, descriptorSetLayout)

  -- Assign the buffer in this descriptor set
  let bufferStruct = zero
        { dstSet = descriptorSet
        , dstBinding = 0
        , descriptorType = DESCRIPTOR_TYPE_STORAGE_BUFFER
        , descriptorCount = 1
        , bufferInfo = [DescriptorBufferInfo buffer 0 WHOLE_SIZE]
        }
  updateDescriptorSets device [SomeStruct bufferStruct] []

  shader <- createShader
  (_, pipelineLayout) <- withPipelineLayout
    device zero { setLayouts = [descriptorSetLayout] } Nothing allocate
  let pipelineCreateInfo = zero { layout = pipelineLayout, stage = shader }
  (_, (_, [computePipeline])) <- withComputePipelines
    device zero [SomeStruct pipelineCreateInfo] Nothing allocate

  oneshot 1e9 $ \commandBuffer -> do
    cmdBindPipeline commandBuffer PIPELINE_BIND_POINT_COMPUTE computePipeline
    cmdBindDescriptorSets commandBuffer PIPELINE_BIND_POINT_COMPUTE pipelineLayout 0 [descriptorSet] []
    cmdDispatch
      commandBuffer
      (ceiling (realToFrac width / realToFrac workgroupX :: Float))
      (ceiling (realToFrac height / realToFrac workgroupY :: Float))
      1
  invalidateAllocation allocator bufferAllocation 0 WHOLE_SIZE

  -- TODO: speed this bit up, it's hopelessly slow
  let pixelAddr :: Int -> Int -> Ptr (FixedArray 4 Float)
      pixelAddr x y = plusPtr (mappedData bufferAllocationInfo)
                              (((y * width) + x) * 4 * sizeOf (0 :: Float))
  liftIO $ JP.withImage width height $ \x y -> do
    let ptr = pixelAddr x y
    [r, g, b, a] <- fmap (\f -> round (f * 255)) <$> peekArray 4 (lowerArrayPtr ptr)
    pure $ JP.PixelRGBA8 r g b a

createShader :: Vulkan (SomeStruct PipelineShaderStageCreateInfo)
createShader = do
  device <- asks ghDevice
  let compCode = $(compileShaderQ Nothing "comp" [glslFile|shaders/julia.comp|])
  (_, compModule) <- withShaderModule device zero { code = compCode } Nothing allocate
  pure $ SomeStruct zero
    { stage = SHADER_STAGE_COMPUTE_BIT
    , module' = compModule
    , name = "main"
    }
