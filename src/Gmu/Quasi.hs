module Gmu.Quasi
  ( compileShaderQ
  , glslFile
  ) where

import Language.Haskell.TH.Quote (quoteFile, QuasiQuoter)
import Vulkan.Utils.ShaderQQ (compileShaderQ, glsl)

glslFile :: QuasiQuoter
glslFile = quoteFile glsl
