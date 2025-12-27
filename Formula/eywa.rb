class Eywa < Formula
  desc "Personal knowledge base with local embeddings and semantic search"
  homepage "https://github.com/ShankarKakumani/eywa"
  version "0.1.0"
  license "MIT"

  on_macos do
    if Hardware::CPU.arm?
      url "https://github.com/ShankarKakumani/eywa/releases/download/v#{version}/eywa-darwin-arm64"
      sha256 "8f24be081b2094b04b55e675c91121ab7d46808df0517d55432a10aa66f10c53" # arm64

      def install
        bin.install "eywa-darwin-arm64" => "eywa"
      end
    else
      url "https://github.com/ShankarKakumani/eywa/releases/download/v#{version}/eywa-darwin-x64"
      sha256 "b4a0ec1f20db5976c56297d5c13b2680f5bef874f09195b705241af157546672" # x64

      def install
        bin.install "eywa-darwin-x64" => "eywa"
      end
    end
  end

  on_linux do
    if Hardware::CPU.arm?
      url "https://github.com/ShankarKakumani/eywa/releases/download/v#{version}/eywa-linux-arm64"
      sha256 "f17c25dc7675dd342d08a1e629fbbc6269037fadb3830405602b4a7742b76cb3" # linux-arm64

      def install
        bin.install "eywa-linux-arm64" => "eywa"
      end
    else
      url "https://github.com/ShankarKakumani/eywa/releases/download/v#{version}/eywa-linux-x64"
      sha256 "e2c2327d539602f4a5034e3ace889e746d6dfd1625830af3eb415310f19ae661" # linux-x64

      def install
        bin.install "eywa-linux-x64" => "eywa"
      end
    end
  end

  test do
    system "#{bin}/eywa", "info"
  end
end
