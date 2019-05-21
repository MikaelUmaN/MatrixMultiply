namespace ManagedMultiply
open System.Threading.Tasks
open System.Threading.Tasks
module DivideConquerParallelTask =

    // Recursively computes matrix matrix multiply. Stops recursion at size T (should fit into L1 cache)
    let divideAndConquer  (A: float[]) (B: float[]) (C: float[]) (N: int) (T: int) =
        let rec divConquer (A: float[]) (B: float[]) (C: float[]) (N: int) (T: int) (n: int) (cr: int) (cc: int) (ar: int) (ac: int) (br: int) (bc: int) =
            if (n <= T) then
                // Base case, standard naive multiply
                ThreeLoops.multFlat A B C N n cr cc ar ac br bc
            else
                // Assume square matrix of size N, divisible by two.
                // Divide A,B,C into four blocks each.
                let n2 = n / 2

                //c00 = a00*b00 + a01*b10
                let c00 = Task.Factory.StartNew(fun () ->
                    divConquer A B C N T n2 cr cc ar ac br bc
                    divConquer A B C N T n2 cr cc ar (ac+n2) (br+n2) bc
                )

                //c01 = a00*b01 + a01*b11
                let c01 = Task.Factory.StartNew(fun () ->
                    divConquer A B C N T n2 cr (cc+n2) ar ac br (bc+n2)
                    divConquer A B C N T n2 cr (cc+n2) ar (ac+n2) (br+n2) (bc+n2)
                )

                //c10 = a10*b00 + a11*b10
                let c10 = Task.Factory.StartNew(fun () ->
                    divConquer A B C N T n2 (cr+n2) cc (ar+n2) ac br bc
                    divConquer A B C N T n2 (cr+n2) cc (ar+n2) (ac+n2) (br+n2) bc
                )

                //c11 = a10*b01 + a11*b11
                let c11 = Task.Factory.StartNew(fun () ->
                    divConquer A B C N T n2 (cr+n2) (cc+n2) (ar+n2) ac br (bc+n2)
                    divConquer A B C N T n2 (cr+n2) (cc+n2) (ar+n2) (ac+n2) (br+n2) (bc+n2)
                )

                let tasks = [| c00; c01; c10; c11 |]
                Task.WaitAll(tasks)

        divConquer A B C N T N 0 0 0 0 0 0