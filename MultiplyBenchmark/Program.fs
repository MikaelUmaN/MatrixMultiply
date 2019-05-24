open System

open BenchmarkDotNet.Running
open BenchmarkDotNet.Configs
open BenchmarkDotNet.Reports
open BenchmarkDotNet.Columns
open BenchmarkDotNet.Horology
open BenchmarkDotNet.Exporters.Csv
open BenchmarkDotNet.Jobs
open BenchmarkDotNet.Toolchains.CsProj

[<EntryPoint>]
let main argv =
    let conf = ManualConfig.Create(DefaultConfig.Instance)
    let summaryStyle = SummaryStyle(true, SizeUnit.KB, TimeUnit.Millisecond, false)
    conf.Add(CsvExporter(CsvSeparator.CurrentCulture, summaryStyle))
    conf.Add(Job.Core.With(CsProjCoreToolchain.NetCoreApp30))

    //BenchmarkRunner.Run<DivideConquerBenchmark.Benchmark>(conf) |> ignore
    BenchmarkRunner.Run<DivideConquerParallelBenchmark.Benchmark>(conf) |> ignore
    //BenchmarkRunner.Run<TiledBenchmark.Benchmark>(conf) |> ignore
    //BenchmarkRunner.Run<ThreeLoopsBenchmark.Benchmark>(conf) |> ignore

    0
