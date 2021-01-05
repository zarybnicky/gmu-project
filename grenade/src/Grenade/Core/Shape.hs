{-# LANGUAGE CPP                   #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE KindSignatures        #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE StandaloneDeriving    #-}
{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE UndecidableInstances  #-}
{-|
Module      : Grenade.Core.Shape
Description : Dependently typed shapes of data which are passed between layers of a network
Copyright   : (c) Huw Campbell, 2016-2017
License     : BSD2
Stability   : experimental


-}
module Grenade.Core.Shape (
    S (..)
  , Shape (..)
  , SShape (..)

  , randomOfShape
  , fromStorable
  ) where

import           Control.DeepSeq (NFData (..))
import           Control.Monad.Random ( MonadRandom, getRandom )

import           Data.Kind (Type)
import           Data.Proxy
import           Data.Serialize
import           Data.Singletons
import           Data.Singletons.TypeLits
import           Data.Vector.Storable ( Vector )
import qualified Data.Vector.Storable as V

import           GHC.TypeLits hiding (natVal)

import qualified Numeric.LinearAlgebra.Static as H
import           Numeric.LinearAlgebra.Static
import qualified Numeric.LinearAlgebra as NLA

import Grenade.OpenCL.Context (mkBufferR, mkBufferL, unsafeWithCL, CLBuffer)

-- | The current shapes we accept.
--   at the moment this is just one, two, and three dimensional
--   Vectors/Matricies.
--
--   These are only used with DataKinds, as Kind `Shape`, with Types 'D1, 'D2, 'D3.
data Shape
  = D1 Nat
  -- ^ One dimensional vector
  | D2 Nat Nat
  -- ^ Two dimensional matrix. Row, Column.
  | D3 Nat Nat Nat
  -- ^ Three dimensional matrix. Row, Column, Channels.

  | CL1 Nat
  | CL2 Nat Nat

-- | Concrete data structures for a Shape.
--
--   All shapes are held in contiguous memory.
--   3D is held in a matrix (usually row oriented) which has height depth * rows.
data S (n :: Shape) where
  S1D :: KnownNat len => R len -> S ('D1 len)
  S2D :: (KnownNat r, KnownNat c) => L r c -> S ('D2 r c)
  S3D :: (KnownNat r, KnownNat c, KnownNat d, KnownNat (r * d)) => L (r * d) c -> S ('D3 r c d)

  S1CL :: KnownNat len => CLBuffer (R len) -> S ('CL1 len)
  S2CL :: (KnownNat r, KnownNat c) => CLBuffer (L r c) -> S ('CL2 r c)

deriving instance Show (S n)

-- Singleton instances.
--
-- These could probably be derived with template haskell, but this seems
-- clear and makes adding the KnownNat constraints simple.
-- We can also keep our code TH free, which is great.
type instance Sing = SShape

data SShape :: Shape -> Type where
  D1Sing :: Sing a -> SShape ('D1 a)
  D2Sing :: Sing a -> Sing b -> SShape ('D2 a b)
  D3Sing :: KnownNat (a * c) => Sing a -> Sing b -> Sing c -> SShape ('D3 a b c)

  CL1Sing :: Sing a -> SShape ('CL1 a)
  CL2Sing :: Sing a -> Sing b -> SShape ('CL2 a b)

instance KnownNat a => SingI ('D1 a) where
  sing = D1Sing sing
instance (KnownNat a, KnownNat b) => SingI ('D2 a b) where
  sing = D2Sing sing sing
instance (KnownNat a, KnownNat b, KnownNat c, KnownNat (a * c)) => SingI ('D3 a b c) where
  sing = D3Sing sing sing sing

instance KnownNat a => SingI ('CL1 a) where
  sing = CL1Sing sing
instance (KnownNat a, KnownNat b) => SingI ('CL2 a b) where
  sing = CL2Sing sing sing

instance SingI x => Num (S x) where
  (+) = n2 (+)
  (-) = n2 (-)
  (*) = n2 (*)
  abs = n1 abs
  signum = n1 signum
  fromInteger x = nk (fromInteger x)

instance SingI x => Fractional (S x) where
  (/) = n2 (/)
  recip = n1 recip
  fromRational x = nk (fromRational x)

instance SingI x => Floating (S x) where
  pi = nk pi
  exp = n1 exp
  log = n1 log
  sqrt = n1 sqrt
  (**) = n2 (**)
  logBase = n2 logBase
  sin = n1 sin
  cos = n1 cos
  tan = n1 tan
  asin = n1 asin
  acos = n1 acos
  atan = n1 atan
  sinh = n1 sinh
  cosh = n1 cosh
  tanh = n1 tanh
  asinh = n1 asinh
  acosh = n1 acosh
  atanh = n1 atanh

--
-- I haven't made shapes strict, as sometimes they're not needed
-- (the last input gradient back for instance)
--
instance NFData (S x) where
  rnf (S1D x) = rnf x
  rnf (S2D x) = rnf x
  rnf (S3D x) = rnf x
  rnf (S1CL x) = rnf x
  rnf (S2CL x) = rnf x

-- | Generate random data of the desired shape
randomOfShape :: forall x m. ( MonadRandom m, SingI x ) => m (S x)
randomOfShape = do
  seed :: Int <- getRandom
  return $ case (sing :: Sing x) of
    D1Sing SNat ->
        S1D (randomVector seed Uniform * 2 - 1)

    D2Sing SNat SNat ->
        S2D (uniformSample seed (-1) 1)

    D3Sing SNat SNat SNat ->
        S3D (uniformSample seed (-1) 1)

    CL1Sing SNat ->
        S1CL $ unsafeWithCL $ \cl -> mkBufferR cl (randomVector seed Uniform * 2 - 1)

    CL2Sing SNat SNat ->
        S2CL $ unsafeWithCL $ \cl -> mkBufferL cl (uniformSample seed (-1) 1)

-- | Generate a shape from a Storable Vector.
--
--   Returns Nothing if the vector is of the wrong size.
fromStorable :: forall x. SingI x => Vector Double -> Maybe (S x)
fromStorable xs = case sing :: Sing x of
    D1Sing SNat ->
      S1D <$> H.create xs

    D2Sing SNat SNat ->
      S2D <$> mkL xs

    D3Sing SNat SNat SNat ->
      S3D <$> mkL xs

    CL1Sing SNat -> error "fromStorable: CL1Sing SNat"
    CL2Sing SNat SNat -> error "fromStorable: CL2Sing SNat SNat"
  where
    mkL :: forall rows columns. (KnownNat rows, KnownNat columns)
        => Vector Double -> Maybe (L rows columns)
    mkL v =
      let rows    = fromIntegral $ natVal (Proxy :: Proxy rows)
          columns = fromIntegral $ natVal (Proxy :: Proxy columns)
      in  if rows * columns == V.length v
             then H.create $ NLA.reshape columns v
             else Nothing


instance SingI x => Serialize (S x) where
  put i = (case i of
            (S1D x) -> putListOf put . NLA.toList . H.extract $ x
            (S2D x) -> putListOf put . NLA.toList . NLA.flatten . H.extract $ x
            (S3D x) -> putListOf put . NLA.toList . NLA.flatten . H.extract $ x
            (S1CL _) -> error "put: S1CL"
            (S2CL _) -> error "put: S2CL"
          ) :: PutM ()

  get = do
    Just i <- fromStorable . V.fromList <$> getListOf get
    return i

-- Helper function for creating the number instances
n1 :: ( forall a. Floating a => a -> a ) -> S x -> S x
n1 f (S1D x) = S1D (f x)
n1 f (S2D x) = S2D (f x)
n1 f (S3D x) = S3D (f x)
n1 _ (S1CL _) = error "Num a => CLBuffer (CL1 n)"
n1 _ (S2CL _) = error "Num a => CLBuffer (CL2 m n)"

-- Helper function for creating the number instances
n2 :: ( forall a. Floating a => a -> a -> a ) -> S x -> S x -> S x
n2 f (S1D x) (S1D y) = S1D (f x y)
n2 f (S2D x) (S2D y) = S2D (f x y)
n2 f (S3D x) (S3D y) = S3D (f x y)
n2 _ (S1CL _) (S1CL _) = error "Num a => CLBuffer (CL1 n)"
n2 _ (S2CL _) (S2CL _) = error "Num a => CLBuffer (CL2 m n)"

-- Helper function for creating the number instances
nk :: forall x. SingI x => Double -> S x
nk x = case (sing :: Sing x) of
  D1Sing SNat ->
    S1D (konst x)

  D2Sing SNat SNat ->
    S2D (konst x)

  D3Sing SNat SNat SNat ->
    S3D (konst x)

  CL1Sing SNat ->
    unsafeWithCL $ \cl -> S1CL <$> mkBufferR cl (konst x)

  CL2Sing SNat SNat ->
    unsafeWithCL $ \cl -> S2CL <$> mkBufferL cl (konst x)
