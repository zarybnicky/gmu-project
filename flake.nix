{
  inputs.nixpkgs.url = github:nixos/nixpkgs/master;
  inputs.CLUtil = { url = github:acowley/CLUtil/master; flake = false; };

  outputs = { self, nixpkgs, CLUtil }: let
    inherit (nixpkgs.lib) flip mapAttrs mapAttrsToList;
    inherit (pkgs.nix-gitignore) gitignoreSourcePure;

    pkgs = import nixpkgs {
      system = "x86_64-linux";
      config.allowUnfree = true;
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
          grenade = hself.callCabal2nix "grenade" (getSrc ./grenade) {};
          CLUtil = dontHaddock (dontCheck (hself.callCabal2nix "CLUtil" CLUtil {}));
          OpenCL = overrideCabal (dontCheck (doJailbreak (unmarkBroken hsuper.OpenCL))) (drv: {
            configureFlags = (drv.configureFlags or []) ++ [
              "--extra-lib-dirs=${pkgs.ocl-icd}/lib"
              "--extra-include-dirs=${pkgs.opencl-headers}/include"
            ];
          });
          profiteur = doJailbreak (unmarkBroken hsuper.profiteur);
        });
      };
    };

    devShell.x86_64-linux = hsPkgs.shellFor {
      withHoogle = true;
      packages = p: [ p.grenade ];
      buildInputs = [
        hsPkgs.cabal-install
        hsPkgs.haskell-language-server
        hsPkgs.stylish-haskell
        hsPkgs.profiteur
      ];
    };
  };
}
