{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE UnliftedNewtypes #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE UnboxedTuples #-}
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE CPP #-}

module Grenade.OpenCL.Context
  ( globalCLState
  , withCL
  , unsafeWithCL
  , descendOpenCL
  , mkBufferL
  , mkBufferR
  , readBufferR
  , readBufferL
  , writeBufferR
  , CLState(..)
  , CLBuffer(..)
  , clEnqueueNDRangeKernel'
  , clSetKernelArgSto'
  , clEnqueueReadBuffer'
  , clEnqueueWriteBuffer'
  ) where

import Control.Parallel.OpenCL
import Control.Exception (throwIO, throw, handle)
import Data.FileEmbed (embedStringFile, makeRelativeToProject)
import Data.IORef (IORef, readIORef, newIORef)
import Data.Maybe (fromJust)
import Data.Proxy (Proxy(Proxy))
import Data.Singletons.TypeLits (KnownNat, natVal)
import Data.Vector.Storable (Vector, unsafeFromForeignPtr0, unsafeToForeignPtr0)
import Foreign.C.Types (CBool(..), CSize(..), CDouble)
import Foreign.Marshal (fromBool, with, alloca, withArray)
import Foreign.ForeignPtr (mallocForeignPtrArray, withForeignPtr)
import Foreign.Ptr (Ptr, castPtr, nullPtr)
import Foreign.Storable (Storable(sizeOf))
import Numeric.LinearAlgebra.Static (L, R, extract, create)
import Numeric.LinearAlgebra.Data (tr, flatten)
import System.IO.Unsafe (unsafePerformIO)
import System.Mem.Weak (addFinalizer)
import Numeric.LinearAlgebra.Devel (MatrixOrder(ColumnMajor), matrixFromVector)
import Data.Int (Int32)
import Data.Word (Word32)

data CLState = CLState
  { clContext :: {-# UNPACK #-} !CLContext
  , clDevice :: {-# UNPACK #-} !CLDeviceID
  , clQueue :: {-# UNPACK #-} !CLCommandQueue
  , clProgram :: {-# UNPACK #-} !CLProgram
  , clKDescend :: {-# UNPACK #-} !CLKernel

  , clWeights :: {-# UNPACK #-} !CLMem
  , clGradient :: {-# UNPACK #-} !CLMem
  , clLastUpdate :: {-# UNPACK #-} !CLMem
  , clNewKernel :: {-# UNPACK #-} !CLMem
  , clNewMomentum :: {-# UNPACK #-} !CLMem
  }

source :: String
source = $(embedStringFile =<< makeRelativeToProject "src-gmu/kernel.cl")

globalCLState :: IORef CLState
{-# NOINLINE globalCLState #-}
globalCLState = unsafePerformIO $ do
  clContext <- clCreateContextFromType [] [CL_DEVICE_TYPE_CPU] print
  clDevice <- head <$> clGetContextDevices clContext
  clQueue <- clCreateCommandQueue clContext clDevice []
  putStrLn . ("Platform: " <>) =<< (`clGetPlatformInfo` CL_PLATFORM_NAME) =<< clGetDevicePlatform clDevice
  putStrLn . ("Device: " <>) =<< clGetDeviceName clDevice
  putStrLn . ("Max work group size: " <>) . show =<< clGetDeviceMaxWorkGroupSize clDevice
  putStrLn . ("Max work item  dims: " <>) . show =<< clGetDeviceMaxWorkItemDimensions clDevice
  putStrLn . ("Max work item sizes: " <>) . show =<< clGetDeviceMaxWorkItemSizes clDevice

  clProgram <- clCreateProgramWithSource clContext source
  handle @CLError
    (\err -> (putStrLn =<< clGetProgramBuildLog clProgram clDevice) >> throw err)
    (clBuildProgram clProgram [clDevice] "")

  clWeights <- clCreateBuffer clContext [CL_MEM_READ_ONLY, CL_MEM_ALLOC_HOST_PTR]
    (sizeOf (undefined :: CDouble) * 500, nullPtr)
  clGradient <- clCreateBuffer clContext [CL_MEM_READ_ONLY, CL_MEM_ALLOC_HOST_PTR]
    (sizeOf (undefined :: CDouble) * 500, nullPtr)
  clLastUpdate <- clCreateBuffer clContext [CL_MEM_READ_ONLY, CL_MEM_ALLOC_HOST_PTR]
    (sizeOf (undefined :: CDouble) * 500, nullPtr)
  clNewKernel <- clCreateBuffer clContext [CL_MEM_WRITE_ONLY, CL_MEM_ALLOC_HOST_PTR]
    (sizeOf (undefined :: CDouble) * 500, nullPtr)
  clNewMomentum <- clCreateBuffer clContext [CL_MEM_WRITE_ONLY, CL_MEM_ALLOC_HOST_PTR]
    (sizeOf (undefined :: CDouble) * 500, nullPtr)

  clKDescend <- clCreateKernel clProgram "descend_slow"
  clSetKernelArgSto clKDescend 3 clWeights
  clSetKernelArgSto clKDescend 4 clGradient
  clSetKernelArgSto clKDescend 5 clLastUpdate
  clSetKernelArgSto clKDescend 6 clNewKernel
  clSetKernelArgSto clKDescend 7 clNewMomentum

  let s = CLState {..}
  addFinalizer s (() <$ clReleaseContext clContext)
  newIORef CLState {..}

unsafeWithCL :: (CLState -> IO a) -> a
{-# INLINE unsafeWithCL #-}
unsafeWithCL = unsafePerformIO . withCL

withCL :: (CLState -> IO a) -> IO a
{-# INLINE withCL #-}
withCL f = f =<< readIORef globalCLState

newtype CLBuffer a = CLBuffer CLMem
  deriving newtype (Storable)

writeBufferR :: forall n. KnownNat n => CLState -> CLBuffer (R n) -> R n -> IO ()
writeBufferR cl (CLBuffer buf) vec = do
  withForeignPtr (fst . unsafeToForeignPtr0 $ extract vec) $ \ptr ->
    () <$ clEnqueueWriteBuffer' (clQueue cl) buf True 0 size (castPtr ptr)
  where
    len = fromIntegral (natVal (Proxy @n))
    size = sizeOf (undefined :: CDouble) * len

readBufferR :: forall n. KnownNat n => CLState -> CLBuffer (R n) -> Maybe (R n) -> IO (R n)
readBufferR cl (CLBuffer buf) mVec = do
  fPtr <- case mVec of
    Nothing -> mallocForeignPtrArray len
    Just vec -> pure $ fst (unsafeToForeignPtr0 (extract vec))
  _ <- withForeignPtr fPtr $ \ptr ->
    clEnqueueReadBuffer' (clQueue cl) buf True 0 size (castPtr ptr)
  pure (fromJust . create $ unsafeFromForeignPtr0 fPtr len)
  where
    len = fromIntegral (natVal (Proxy @n))
    size = sizeOf (undefined :: CDouble) * len

readBufferL :: forall m n. (KnownNat m, KnownNat n) => CLState -> CLBuffer (L m n) -> Maybe (L m n) -> IO (L m n)
readBufferL cl (CLBuffer buf) mMat = do
  fPtr <- case mMat of
    Nothing -> mallocForeignPtrArray len
    Just mat -> pure $ fst (unsafeToForeignPtr0 (flatten . tr $ extract mat))
  _ <- withForeignPtr fPtr $ \ptr ->
    clEnqueueReadBuffer' (clQueue cl) buf True 0 size (castPtr ptr)
  pure (fromJust . create . matrixFromVector ColumnMajor rows cols $ unsafeFromForeignPtr0 fPtr len)
  where
    rows = fromIntegral (natVal (Proxy @m))
    cols = fromIntegral (natVal (Proxy @n))
    len = rows * cols
    size = sizeOf (undefined :: CDouble) * len

mkBufferR :: forall n. KnownNat n => CLState -> R n -> IO (CLBuffer (R n))
mkBufferR cl vec =
  withForeignPtr (fst (unsafeToForeignPtr0 (extract vec))) $ \ptr ->
    CLBuffer <$> clCreateBuffer (clContext cl) [CL_MEM_READ_WRITE, CL_MEM_COPY_HOST_PTR] (size, castPtr ptr)
  where
    len = fromIntegral (natVal (Proxy @n))
    size = sizeOf (undefined :: CDouble) * len

mkBufferL :: forall m n. (KnownNat m, KnownNat n) => CLState -> L m n -> IO (CLBuffer (L m n))
mkBufferL cl mat =
  withForeignPtr (fst (unsafeToForeignPtr0 vec)) $ \ptr -> do
    buf <- clCreateBuffer (clContext cl) [CL_MEM_READ_WRITE, CL_MEM_COPY_HOST_PTR] (size, castPtr ptr)
    pure $ CLBuffer buf
  where
    vec = flatten (tr (extract mat))
    len = fromIntegral (natVal (Proxy @m)) * fromIntegral (natVal (Proxy @n))
    size = sizeOf (undefined :: CDouble) * len

descendOpenCL
  :: Int
  -> Double
  -> Double
  -> Double
  -> Vector Double
  -> Vector Double
  -> Vector Double
  -> (Vector Double, Vector Double)
descendOpenCL len rate momentum regulariser weights gradient lastUpdate =
  unsafeWithCL $ \cl -> do
    outWPtr <- mallocForeignPtrArray len
    outMPtr <- mallocForeignPtrArray len
    let (wPtr, _) = unsafeToForeignPtr0 weights
    let (gPtr, _) = unsafeToForeignPtr0 gradient
    let (lPtr, _) = unsafeToForeignPtr0 lastUpdate

    withForeignPtr wPtr $ \wPtr' ->
      withForeignPtr gPtr $ \gPtr' ->
      withForeignPtr lPtr $ \lPtr' ->
      withForeignPtr outWPtr $ \outWPtr' ->
      withForeignPtr outMPtr $ \outMPtr' -> do
        let l = len * sizeOf (undefined :: CDouble)
        es <- sequence
          [ clEnqueueWriteBuffer (clQueue cl) (clWeights cl) True 0 l (castPtr wPtr') []
          , clEnqueueWriteBuffer (clQueue cl) (clGradient cl) True 0 l (castPtr gPtr') []
          , clEnqueueWriteBuffer (clQueue cl) (clLastUpdate cl) True 0 l (castPtr lPtr') []
          ]
        clSetKernelArgSto (clKDescend cl) 0 rate
        clSetKernelArgSto (clKDescend cl) 1 momentum
        clSetKernelArgSto (clKDescend cl) 2 regulariser
        e <- clEnqueueNDRangeKernel (clQueue cl) (clKDescend cl) [len] [] es
        _ <- clWaitForEvents =<< sequence
          [ clEnqueueReadBuffer (clQueue cl) (clNewKernel cl) True 0 l (castPtr outWPtr') [e]
          , clEnqueueReadBuffer (clQueue cl) (clNewMomentum cl) True 0 l (castPtr outMPtr') [e]
          ]
        pure (unsafeFromForeignPtr0 outWPtr len, unsafeFromForeignPtr0 outMPtr len)


clEnqueueReadBuffer' :: Integral a => CLCommandQueue -> CLMem -> Bool -> a -> a
                       -> Ptr () -> IO Int32
clEnqueueReadBuffer' cq mem check off size dat =
  alloca $ \event -> do
    _ <- raw_clEnqueueReadBuffer cq mem (fromBool check) (fromIntegral off) (fromIntegral size) dat 0 nullPtr event
    raw_clWaitForEvents 1 (castPtr event)

clEnqueueWriteBuffer' :: Integral a => CLCommandQueue -> CLMem -> Bool -> a -> a
                       -> Ptr () -> IO Int32
clEnqueueWriteBuffer' cq mem check off size dat =
  alloca $ \event -> do
    _ <- raw_clEnqueueWriteBuffer cq mem (fromBool check) (fromIntegral off) (fromIntegral size) dat 0 nullPtr event
    raw_clWaitForEvents 1 (castPtr event)

clEnqueueNDRangeKernel' :: CLCommandQueue -> CLKernel -> [CSize] -> IO Int32
clEnqueueNDRangeKernel' cq krn gws =
  withArray gws $ \pgws ->
  alloca $ \event ->
    whenSuccess
      (raw_clEnqueueNDRangeKernel cq krn num nullPtr pgws nullPtr 0 nullPtr event)
      (raw_clWaitForEvents 1 (castPtr event))
  where
    num = fromIntegral $ length gws

clSetKernelArgSto' :: Storable a => CLKernel -> Word32 -> a -> IO ()
clSetKernelArgSto' krn idx val = with val $ \pval -> do
  () <$ raw_clSetKernelArg krn idx (fromIntegral . sizeOf $ val) (castPtr pval)

whenSuccess :: IO Int32 -> IO a -> IO a
{-# INLINE whenSuccess #-}
whenSuccess fcheck fval = do
  errcode <- toEnum . fromIntegral <$> fcheck
  if errcode == CL_SUCCESS
    then fval
    else throwIO errcode

foreign import ccall "clEnqueueNDRangeKernel" raw_clEnqueueNDRangeKernel
  :: CLCommandQueue
  -> CLKernel
  -> Word32
  -> Ptr CSize
  -> Ptr CSize
  -> Ptr CSize
  -> Word32
  -> Ptr CLEvent
  -> Ptr CLEvent
  -> IO Int32

foreign import ccall "clEnqueueReadBuffer" raw_clEnqueueReadBuffer
  :: CLCommandQueue
  -> CLMem
  -> CBool
  -> CSize
  -> CSize
  -> Ptr ()
  -> Word32
  -> Ptr CLEvent
  -> Ptr CLEvent
  -> IO Int32

foreign import ccall "clEnqueueWriteBuffer" raw_clEnqueueWriteBuffer
  :: CLCommandQueue
  -> CLMem
  -> CBool
  -> CSize
  -> CSize
  -> Ptr ()
  -> Word32
  -> Ptr CLEvent
  -> Ptr CLEvent
  -> IO Int32

foreign import ccall "clWaitForEvents" raw_clWaitForEvents
  :: Word32
  -> Ptr CLEvent
  -> IO Int32

foreign import ccall "clSetKernelArg" raw_clSetKernelArg
  :: CLKernel
  -> Word32
  -> CSize
  -> Ptr ()
  -> IO Int32
