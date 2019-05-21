namespace ManagedMultiply
open System

module ThreeLoops =

    let multFlat (A: float[]) (B: float[]) (C: float[]) (N: int) (n: int) (cr: int) (cc: int) (ar: int) (ac: int) (br: int) (bc: int) =
        // We are multiplying a small submatrix of size N by N.
        // The loops are the same but the adressing is adjusted by the row and col indices
        for i in 0..n-1 do
            for j in 0..n-1 do
                let mutable s = 0.
                for k in 0..n-1 do
                    s <- s + A.[(i+ar)*N + k + ac] * B.[(k+br)*N + j + bc]
                C.[(i+cr)*N + j + cc] <- C.[(i+cr)*N + j + cc] + s

    // Row major addressing matrix multiply    
    let mult (A: float[]) (B: float[]) (C: float[]) N =
        for i in [0..N-1] do //for each row in A
            for j in [0..N-1] do //for each column in B
                let mutable s = 0.
                for k in [0..N-1] do //dotproduct row and column
                    s <- s + A.[i*N + k] * B.[k*N + j] //stride of N floats every step
                C.[i*N + j] <- s

    // Row major multiply with a tile size of T
    let tiledMult (A: float[]) (B: float[]) (C: float[]) N T =
        for I in [0..T..N-1] do
            for J in [0..T..N-1] do
                for K in [0..T..N-1] do // Perform regular matrix multiply in blocks of tile size T.
                    for i = I to Math.Min(I+T-1, N-1) do
                        for j = J to Math.Min(J+T-1, N-1) do
                            let mutable s = 0.
                            for k = K to Math.Min(K+T-1, N-1) do
                                s <- s + A.[i*N + k] * B.[k*N + j]
                            C.[i*N + j] <- C.[i*N + j] + s // Cij contributions come from several blocks.
    