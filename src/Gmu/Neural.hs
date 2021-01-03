{-# LANGUAGE TupleSections #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeApplications #-}

module Gmu.Neural
  ( runNn
  ) where

import CLUtil
import Control.Parallel.OpenCL
import Data.FileEmbed (embedStringFile)
import qualified Data.Vector.Storable as V
import Foreign
import Data.Traversable (forM)
import Control.Exception (throw, handle)
import Control.Arrow (Arrow((&&&)))

runNn :: IO ()
runNn = do
  context <- clCreateContextFromType [] [CL_DEVICE_TYPE_CPU] print
  device <- head <$> clGetContextDevices context
  queue <- clCreateCommandQueue context device []

  putStrLn . ("Platform: " <>) =<< (`clGetPlatformInfo` CL_PLATFORM_NAME) =<< clGetDevicePlatform device
  putStrLn . ("Device: " <>) =<< clGetDeviceName device
  putStrLn . ("Max work group size: " <>) . show =<< clGetDeviceMaxWorkGroupSize device
  putStrLn . ("Max work item  dims: " <>) . show =<< clGetDeviceMaxWorkItemDimensions device
  putStrLn . ("Max work item sizes: " <>) . show =<< clGetDeviceMaxWorkItemSizes device

  kernels <- loadKernels context device
  let layers = [mkLayer 3 0, mkLayer 2 3]
  mapM_ print (( id &&& sizeOf) <$> layers)

  (layerBufs, evs) <- fmap unzip . forM layers $ \layer -> do
    let size = sizeOf layer
    buf <- (layer, , size) <$> clCreateBuffer context [CL_MEM_READ_WRITE, CL_MEM_ALLOC_HOST_PTR] (size, nullPtr)
    (buf,) <$> writeVector queue buf []
  putStrLn "Written"
  mapM_ (\buf -> print . fst =<< readVector @Layer queue buf []) layerBufs

  evs' <- forward queue kernels layerBufs evs
  print . fst =<< readVector @Layer queue (layerBufs !! 1) evs
  -- _ <- clWaitForEvents evs'
  -- mapM_ (\buf -> print . fst =<< readVector @Layer queue buf []) layerBufs

  () <$ clReleaseContext context

mkLayer :: Int -> Int -> Layer
mkLayer nodes weights = Layer (fromIntegral nodes) $ V.replicate nodes $
  Node 0 0 (fromIntegral weights) (V.replicate weights 0)

forward :: CLCommandQueue -> CLKernels -> [(Layer, CLMem, Int)] -> [CLEvent] -> IO [CLEvent]
forward queue kernels layers evs =
  forM (zip [0..] (zip layers (drop 1 layers))) $ \(i, ((_, prevBuf, _), (cur, curBuf, _))) -> do
    let kernel = kCompOut kernels
        softmax = i == length layers - 1
    clSetKernelArgSto kernel 0 curBuf
    clSetKernelArgSto kernel 1 prevBuf
    clSetKernelArgSto @Int32 kernel 2 (if softmax then 1 else 0)
    ev <- clEnqueueNDRangeKernel queue kernel [V.length (layerNodes cur)] [] evs
    if softmax
      then do
        let kernel' = kSoftmax kernels
        clSetKernelArgSto kernel' 0 curBuf
        clEnqueueNDRangeKernel queue kernel' [V.length (layerNodes cur)] [] [ev]
      else pure ev

writeVector :: Storable a => CLCommandQueue -> (a, CLMem, Int) -> [CLEvent] -> IO CLEvent
writeVector queue (x, mem, size) evs = do
  (e, dst) <- clEnqueueMapBuffer queue mem True [CL_MAP_WRITE] 0 size evs
  poke (castPtr dst) x
  clEnqueueUnmapMemObject queue mem dst [e]

readVector :: Storable a => CLCommandQueue -> (b, CLMem, Int) -> [CLEvent] -> IO (a, CLEvent)
readVector queue (_, mem, size) evs = do
  (e, dst) <- clEnqueueMapBuffer queue mem True [CL_MAP_READ] 0 size evs
  x <- peek (castPtr dst)
  e' <- clEnqueueUnmapMemObject queue mem dst [e]
  pure (x, e')

data Layer = Layer
  { layerNumNodes :: Int32
  , layerNodes :: V.Vector Node
  } deriving (Show, Eq)

instance Storable Layer where
  sizeOf _ = 4 + 399 * 4
  alignment _ = 4
  peekByteOff ptr off = do
    n <- peekByteOff ptr off
    Layer n <$> V.unfoldrNM
      (fromIntegral n)
      (\(i, j) -> if j < n
        then do
          print (i, j, n, off)
          x <- peekByteOff ptr (off + 4 + i)
          pure $ Just (x, (i + sizeOf x, j + 1))
        else pure Nothing)
      (0, 0)
  pokeByteOff ptr off (Layer a b) = do
    pokeByteOff ptr off a
    () <$ V.foldM (\i x -> (i + 1600) <$ pokeByteOff ptr (off + 4 + i) x) 0 b

data Node = Node
  { nodeOutput :: CFloat
  , nodeDelta :: CFloat
  , nodeNumWeights :: Int32
  , nodeWeights :: V.Vector CFloat
  } deriving (Show, Eq)

instance Storable Node where
  sizeOf _ = 12 + 397 * 4
  alignment _ = 4
  peekByteOff ptr off = do
    n <- peekByteOff ptr (off + 8)
    Node
      <$> peekByteOff ptr off
      <*> peekByteOff ptr (off + 4)
      <*> pure n
      <*> V.generateM (fromIntegral n) (\i -> peekByteOff ptr (off + 12 + i * 4))
  pokeByteOff ptr off (Node a b c d) = do
    pokeByteOff ptr off a
    pokeByteOff ptr (off + 4) b
    pokeByteOff ptr (off + 8) c
    () <$ V.foldM (\i x -> (i + 1) <$ pokeByteOff ptr (off + 12 + i * 4) x) 0 d


loadKernels :: CLContext -> CLDeviceID -> IO CLKernels
loadKernels context device = do
  program <- clCreateProgramWithSource context source
  handle @CLError
    (\err -> (putStrLn =<< clGetProgramBuildLog program device) >> throw err)
    (clBuildProgram program [device] "")
  CLKernels
    <$> clCreateKernel program "compout"
    <*> clCreateKernel program "softmax"
    <*> clCreateKernel program "backprophid"
    <*> clCreateKernel program "backpropout"
  where
    source = $(embedStringFile "kernel.cl") :: String

data CLKernels = CLKernels
  { kCompOut :: CLKernel
  , kSoftmax :: CLKernel
  , kBatchPropHid :: CLKernel
  , kBatchPropOut :: CLKernel
  }
