import { afterAll, describe, expect, test } from "vitest";
import { BinaryOp, LiteralNum, Variable } from "../../../../core/tree";
import { compileTape } from "../../../tape";
import { compileGpuCodegenFromTape, planGpuCodegenChunks } from "./codegen";
import { destroyGpu, initGpu } from "../gpu-util";

const shouldTestGpu = process.env.LONA_TEST_GPU === "1";
const gpuAvailable = shouldTestGpu && (await initGpu()) !== null;
const gpuTest = test.skipIf(!gpuAvailable);

afterAll(async () => {
  if (shouldTestGpu) await destroyGpu();
});

describe("GPU codegen chunk planning", () => {
  test("bounds every chunk by node count", () => {
    let root: Variable | BinaryOp = new Variable("x");
    for (let i = 0; i < 20; i++) {
      root = new BinaryOp("ADD", root, new LiteralNum(i + 1));
    }
    const tape = compileTape([root])!;
    const plan = planGpuCodegenChunks(tape, 3);

    expect(plan.chunks.length).toBeGreaterThan(1);
    expect(plan.chunks.every((c) => c.end - c.start <= 3)).toBe(true);
    expect(plan.chunks[0]!.start).toBe(0);
    expect(plan.chunks[plan.chunks.length - 1]!.end).toBe(tape.opcodes.length);
    expect(plan.numEscapeSlots).toBeGreaterThan(0);
  });

  test("marks a shared value once when it crosses a chunk boundary", () => {
    const x = new Variable("x");
    const shared = new BinaryOp("MUL", x, x);
    const root = new BinaryOp(
      "ADD",
      new BinaryOp("ADD", shared, new LiteralNum(1)),
      shared,
    );
    const tape = compileTape([root])!;
    const plan = planGpuCodegenChunks(tape, 1);
    const sharedIndex = tape.argB[tape.opcodes.length - 1]!;

    expect(plan.escapeSlot[sharedIndex]).toBeGreaterThanOrEqual(0);
    expect(
      Array.from(plan.escapeSlot).filter(
        (slot) => slot === plan.escapeSlot[sharedIndex],
      ),
    ).toHaveLength(1);
  });

  test("rejects invalid chunk limits", () => {
    const tape = compileTape([new LiteralNum(1)])!;
    expect(() => planGpuCodegenChunks(tape, 0)).toThrow(
      /Invalid GPU codegen chunk node limit/,
    );
  });

  gpuTest("evaluates cross-chunk dependencies", async () => {
    const x = new Variable("x");
    const shared = new BinaryOp("MUL", x, x);
    const root = new BinaryOp(
      "ADD",
      new BinaryOp("ADD", shared, new LiteralNum(1)),
      shared,
    );
    const gpu = compileGpuCodegenFromTape(compileTape([root])!, {
      maxChunkNodes: 1,
    });
    expect(gpu.numChunks).toBeGreaterThan(1);
    expect(await gpu.evalBatch(new Float32Array([2, 3]), 2)).toEqual(
      new Float32Array([9, 19]),
    );
    gpu.destroy();
  });
});
