module IntelMkl

open MathNet.Numerics.LinearAlgebra
open BenchmarkDotNet.Attributes
open ManagedMultiply
open MathNet.Numerics.Providers
open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra.Double

type Benchmark() =

    let mutable A: Matrix<float> = null
    let mutable B: Matrix<float> = null
    let mutable C: Matrix<float> = null

    [<Params(4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)>]
    member val N = 0 with get, set

    [<GlobalSetup>]
    member this.Setup() =
        Control.UseNativeMKL()
        A <- DenseMatrix.randomStandard this.N this.N
        B <- DenseMatrix.randomStandard this.N this.N

    [<Benchmark>]
    member this.Benchmark() =
        C <- A * B
        C
