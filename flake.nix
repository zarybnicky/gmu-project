{
  inputs.nixpkgs.url = github:nixos/nixpkgs/master;

  outputs = { self, nixpkgs }: let
    inherit (nixpkgs.lib) flip mapAttrs mapAttrsToList;
    inherit (pkgs.nix-gitignore) gitignoreSourcePure;

    pkgs = import nixpkgs {
      system = "x86_64-linux";
      config.allowUnfree = true;
      overlays = [ self.overlay ];
    };
  in {
    overlay = final: prev: {
      gmu-opencl = pkgs.stdenv.mkDerivation rec {
        name = "gmu-opencl";
        src = gitignoreSourcePure [./.gitignore] ./.;
        configurePhase = "${pkgs.xxd}/bin/xxd -i kernel.cl > kernel.c && cmake .";
        buildPhase = "make";
        installPhase = "mkdir -p $out/bin && mv gmu-nn $out/bin";
        buildInputs = with pkgs; [
          cmake
          boost
          opencl-headers
          hexdump
          rocm-opencl-runtime
          opencv2
        ];
      };
    };

    defaultPackage.x86_64-linux = pkgs.gmu-opencl;
    devShell.x86_64-linux = pkgs.mkShell {
      inputsFrom = [pkgs.gmu-opencl];
      buildInputs = [pkgs.gdb pkgs.valgrind];
    };
  };
}
