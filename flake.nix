{
  inputs.nixpkgs.url = github:nixos/nixpkgs/master;
  inputs.CLUtil = { url = github:acowley/CLUtil/master; flake = false; };

  outputs = { self, nixpkgs, CLUtil }: let
    inherit (nixpkgs.lib) flip mapAttrs mapAttrsToList;
    inherit (pkgs.nix-gitignore) gitignoreSourcePure gitignoreSource;

    pkgs = import nixpkgs {
      system = "x86_64-linux";
      config = {
        allowUnfree = true;
      };
      overlays = [ self.overlay ];
    };
    hsPkgs = pkgs.haskellPackages;
    getSrc = dir: gitignoreSourcePure [./.gitignore] dir;
  in {
    overlay = final: prev: let
      inherit (prev.haskell.lib) doJailbreak dontCheck unmarkBroken
        overrideCabal dontHaddock;
    in {
      haskell = prev.haskell // {
        packageOverrides = prev.lib.composeExtensions (prev.haskell.packageOverrides or (_: _: {})) (hself: hsuper: {
          nn-accelerate-cuda = hself.callCabal2nix "nn-accelerate-cuda" (getSrc ./.) {};
          autoapply = doJailbreak (unmarkBroken hsuper.autoapply);
          CLUtil = dontHaddock (dontCheck (hself.callCabal2nix "CLUtil" CLUtil {}));
          OpenCL = overrideCabal (dontCheck (doJailbreak (unmarkBroken hsuper.OpenCL))) (drv: {
            configureFlags = (drv.configureFlags or []) ++ [
              "--extra-lib-dirs=${pkgs.ocl-icd}/lib"
              "--extra-include-dirs=${pkgs.opencl-headers}/include"
            ];
          });
        });
      };
    };

    devShell.x86_64-linux = hsPkgs.shellFor {
      withHoogle = true;
      packages = p: [ p.nn-accelerate-cuda ];
      buildInputs = [
        hsPkgs.cabal-install
        hsPkgs.haskell-language-server
        pkgs.shaderc
        hsPkgs.stylish-haskell
      ];
    };

  };
}
