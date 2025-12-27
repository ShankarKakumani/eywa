class Eywa < Formula
  desc "Personal knowledge base with local embeddings and semantic search"
  homepage "https://github.com/ShankarKakumani/eywa"
  version "0.1.2"
  license "MIT"

  on_macos do
    if Hardware::CPU.arm?
      url "https://github.com/ShankarKakumani/eywa/releases/download/v#{version}/eywa-darwin-arm64"
      sha256 "b046109fc575148752c956b2243123602fda85e992ec72316aeca8e9cb94fb95" # arm64

      def install
        bin.install "eywa-darwin-arm64" => "eywa"
      end
    else
      url "https://github.com/ShankarKakumani/eywa/releases/download/v#{version}/eywa-darwin-x64"
      sha256 "cea957339a0ac26d4011eea82491072bbae1f23d2c2f58f7c63087d352ce0a1e" # x64

      def install
        bin.install "eywa-darwin-x64" => "eywa"
      end
    end
  end

  on_linux do
    if Hardware::CPU.arm?
      url "https://github.com/ShankarKakumani/eywa/releases/download/v#{version}/eywa-linux-arm64"
      sha256 "PLACEHOLDER" # linux-arm64

      def install
        bin.install "eywa-linux-arm64" => "eywa"
      end
    else
      url "https://github.com/ShankarKakumani/eywa/releases/download/v#{version}/eywa-linux-x64"
      sha256 "d0c5b722b6935cf2bcb3910450907bfaeb938bfbb19042e40162cee53f426e42" # linux-x64

      def install
        bin.install "eywa-linux-x64" => "eywa"
      end
    end
  end

  test do
    system "#{bin}/eywa", "info"
  end
end
