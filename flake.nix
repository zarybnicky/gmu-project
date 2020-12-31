{
  inputs.nixpkgs.url = github:nixos/nixpkgs/master;
  inputs.vulkan = { url = github:expipiplus1/vulkan/master; flake = false; };

  outputs = { self, nixpkgs, vulkan }: let
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
    vulkanPkgs = import vulkan {
      inherit pkgs;
      forShell = false;
    };
  in {
    overlay = final: prev: let
      inherit (prev.haskell.lib) doJailbreak dontCheck unmarkBroken addExtraLibrary;
    in {
      haskell = prev.haskell // {
        packageOverrides = prev.lib.composeExtensions (prev.haskell.packageOverrides or (_: _: {})) (hself: hsuper: {
          nn-accelerate-cuda = hself.callCabal2nix "nn-accelerate-cuda" (getSrc ./.) {};
          autoapply = doJailbreak (unmarkBroken hsuper.autoapply);
          # inherit (vulkanPkgs)
          #   vulkan vulkan-utils VulkanMemoryAllocator openxr pretty-simple derive-storable
          #   derive-storable-plugin nothunks eventlog2html hs-speedoscope
          #   hvega pandoc language-c;
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
