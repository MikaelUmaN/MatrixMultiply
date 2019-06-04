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



    //KernelTest.nopTest 8

    let conf = ManualConfig.Create(DefaultConfig.Instance)
    let summaryStyle = SummaryStyle(true, SizeUnit.KB, TimeUnit.Millisecond, false)
    conf.Add(CsvExporter(CsvSeparator.CurrentCulture, summaryStyle))
    conf.Add(Job.Clr.With(CsProjClassicNetToolchain.Net472))

    BenchmarkRunner.Run<Nop.Benchmark>(conf) |> ignore
    BenchmarkRunner.Run<NaiveGpuMult.Benchmark>(conf) |> ignore
    BenchmarkRunner.Run<CublasMult.Benchmark>(conf) |> ignore

    0