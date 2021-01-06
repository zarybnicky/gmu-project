{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE TypeOperators         #-}

module Grenade.OpenCL.FullyConnected (
    FullyConnectedCL (..)
  , FullyConnected'CL (..)
  , randomFullyConnectedCL
  ) where

import Control.Monad.Random hiding (fromList)
import Control.Parallel.OpenCL (CLKernel, clCreateKernel)
import Data.Int (Int32)
import Data.Proxy (Proxy(Proxy))
import Data.Singletons.TypeLits
import Grenade.Core
import Grenade.OpenCL.Context
import Numeric.LinearAlgebra.Static (R, L, konst, uniformSample, randomVector, RandDist(..))
import Data.IORef (writeIORef, readIORef, newIORef, IORef)

-- | A basic fully connected (or inner product) neural network layer.
data FullyConnectedCL i o = FullyConnectedCL
  { kUpdate :: !CLKernel
  , kForward :: !CLKernel
  , kBackward :: !CLKernel
  , lpRef :: !(IORef (Maybe LearningParameters))
  , wIn :: !(CLBuffer (R i))          -- work vector
  , wOut :: !(CLBuffer (R o))          -- work vector
  , vecTape :: !(CLBuffer (R i))          -- work vector
  , vecGradient :: !(FullyConnected'CL i o)   -- work gradient
  , vecWeights :: !(FullyConnected'CL i o)   -- Neuron weights
  , vecMomentum :: !(FullyConnected'CL i o)   -- Neuron momentum
  }

data FullyConnected'CL i o = FullyConnected'CL
  !(CLBuffer (R o))   -- Bias
  !(CLBuffer (L o i)) -- Activations

instance Show (FullyConnectedCL i o) where
  show FullyConnectedCL {} = "FullyConnectedCL"

instance (KnownNat i, KnownNat o) => UpdateLayer (FullyConnectedCL i o) where
  type Gradient (FullyConnectedCL i o) = ()

  --   let (nB, nBM) = descendVector lp oB bG oBM
  --       (nA, nM) = descendMatrix lp oA aG oM
  --   in FullyConnectedCL (FullyConnected'CL nB nA) (FullyConnected'CL nBM nM)
  runUpdate lp n () =
    unsafeWithCL $ \cl -> do
      let k = kUpdate n
      lastLp <- readIORef (lpRef n)
      unless (Just lp == lastLp) $ do
        clSetKernelArgSto' k 2 (learningRate lp)
        clSetKernelArgSto' k 3 (learningMomentum lp)
        clSetKernelArgSto' k 4 (learningRegulariser lp)
        writeIORef (lpRef n) (Just lp)
      _ <- clEnqueueNDRangeKernel' (clQueue cl) k [len]
      pure n
    where
      len = fromIntegral (natVal (Proxy @o))

  createRandom = randomFullyConnectedCL

instance (KnownNat i, KnownNat o) => Layer (FullyConnectedCL i o) ('D1 i) ('D1 o) where
  type Tape (FullyConnectedCL i o) ('D1 i) ('D1 o) = ()

  -- Do a matrix vector multiplication and return the result.
  -- (v, S1D (wB + wN #> v))
  runForwards n (S1D v) =
    unsafeWithCL $ \cl -> do
      writeBufferR cl (vecTape n) v
      _ <- clEnqueueNDRangeKernel' (clQueue cl) (kForward n) [len]
      res <- readBufferR cl (wOut n) (Just (konst 0))
      pure ((), S1D res)
    where
      len = fromIntegral (natVal (Proxy @o))

  -- Run a backpropogation step for a full connected layer.
  -- let wB' = dEdy; mm' = dEdy `outer` x
  --     dWs  = tr wN #> dEdy
  -- in  (FullyConnected'CL wB' mm', S1D dWs)
  runBackwards n () (S1D dEdy) =
    unsafeWithCL $ \cl -> do
      let FullyConnected'CL bG _ = vecGradient n
      writeBufferR cl bG dEdy
      _ <- clEnqueueNDRangeKernel' (clQueue cl) (kBackward n) [len]
      dWs <- readBufferR cl (wIn n) (Just (konst 0))
      pure ((), S1D dWs)
    where
      len = fromIntegral (max (natVal (Proxy @o)) (natVal (Proxy @i)))

randomFullyConnectedCL
  :: forall i o m. (MonadIO m, MonadRandom m, KnownNat i, KnownNat o)
  => m (FullyConnectedCL i o)
randomFullyConnectedCL = do
  s1 <- getRandom
  s2 <- getRandom
  liftIO . withCL $ \cl -> do
    oB <- mkBufferR cl (randomVector s1 Uniform * 2 - 1)
    oA <- mkBufferL cl (uniformSample s2 (-1) 1)
    oBM <- mkBufferR cl (konst 0)
    oM <- mkBufferL cl (konst 0)
    bG <- mkBufferR cl (konst 0)
    aG <- mkBufferL cl (konst 0)
    wIn <- mkBufferR cl (konst 0)
    wOut <- mkBufferR cl (konst 0)
    vecTape <- mkBufferR cl (konst 0)
    let vecWeights = FullyConnected'CL oB oA
        vecMomentum = FullyConnected'CL oBM oM
        vecGradient = FullyConnected'CL bG aG

    kUpdate <- clCreateKernel (clProgram cl) "fcnn_update"
    clSetKernelArgSto' kUpdate 0 (fromIntegral @_ @Int32 (natVal (Proxy @o)))
    clSetKernelArgSto' kUpdate 1 (fromIntegral @_ @Int32 (natVal (Proxy @i)))
    clSetKernelArgSto' kUpdate 5 oB
    clSetKernelArgSto' kUpdate 6 oA
    clSetKernelArgSto' kUpdate 7 bG
    clSetKernelArgSto' kUpdate 8 aG
    clSetKernelArgSto' kUpdate 9 oBM
    clSetKernelArgSto' kUpdate 10 oM

    kForward <- clCreateKernel (clProgram cl) "fcnn_forward"
    clSetKernelArgSto' kForward 0 (fromIntegral @_ @Int32 (natVal (Proxy @o)))
    clSetKernelArgSto' kForward 1 (fromIntegral @_ @Int32 (natVal (Proxy @i)))
    clSetKernelArgSto' kForward 2 oA
    clSetKernelArgSto' kForward 3 vecTape
    clSetKernelArgSto' kForward 4 oB
    clSetKernelArgSto' kForward 5 wOut

    kBackward <- clCreateKernel (clProgram cl) "fcnn_backward"
    clSetKernelArgSto' kBackward 0 (fromIntegral @_ @Int32 (natVal (Proxy @o)))
    clSetKernelArgSto' kBackward 1 (fromIntegral @_ @Int32 (natVal (Proxy @i)))
    clSetKernelArgSto' kBackward 2 bG
    clSetKernelArgSto' kBackward 3 vecTape
    clSetKernelArgSto' kBackward 4 oA
    clSetKernelArgSto' kBackward 5 aG
    clSetKernelArgSto' kBackward 6 wIn

    lpRef <- newIORef Nothing

    pure $ FullyConnectedCL{..}
