module Nop

open BenchmarkDotNet.Attributes

type Benchmark() =

    [<Params(4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)>]
    member val N = 0 with get, set

    [<GlobalSetup>]
    member this.Setup() =
        Alea.Settings.Instance.Resource.AssemblyPath <- @"C:\dev\FSharp\MatrixMultiplyGitHub\GPU\packages\Alea.3.0.4\tools"
        Alea.Settings.Instance.Resource.Path <- @"C:\dev\FSharp\MatrixMultiplyGitHub\GPU\ResPath"

    [<Benchmark>]
    member this.Benchmark() =
        KernelTest.nopTest this.N
        this.N
