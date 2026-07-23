import type { VarName } from "../../../core/tree";
import type { CompiledTape } from "../../tape";
import type {
  JvpKernel,
  KernelEnvelope,
  SyncGradKernel,
  SyncJacobianKernel,
  SyncJvpKernel,
} from "../backend";

function requireSyncJvp(
  envelope: KernelEnvelope<JvpKernel>,
): KernelEnvelope<SyncJvpKernel> {
  if (envelope.kernel.kind !== "sync-jvp") {
    envelope.dispose?.();
    throw new Error(
      `backend '${envelope.backend}' cannot adapt an asynchronous JVP to a synchronous derivative routine`,
    );
  }
  return envelope as KernelEnvelope<SyncJvpKernel>;
}

function identitySeeds(
  tape: CompiledTape,
  diffVars: readonly VarName[],
): Float64Array {
  const seeds = new Float64Array(tape.numVars * diffVars.length);
  for (let input = 0; input < tape.numVars; input++) {
    for (let direction = 0; direction < diffVars.length; direction++) {
      if (tape.varSlots[input] === diffVars[direction]) {
        seeds[input * diffVars.length + direction] = 1;
      }
    }
  }
  return seeds;
}

function packValues(tape: CompiledTape, vars: ReadonlyMap<VarName, number>) {
  return Float64Array.from(
    tape.varSlots.slice(0, tape.numVars),
    (name) => vars.get(name) ?? 0,
  );
}

/** Adapt a seeded, multi-root synchronous JVP to a single-root gradient. */
export function adaptSyncJvpToGrad(
  tape: CompiledTape,
  diffVars: readonly VarName[],
  envelope: KernelEnvelope<JvpKernel>,
): KernelEnvelope<SyncGradKernel> {
  const jvp = requireSyncJvp(envelope);
  if (jvp.kernel.numRoots !== 1) {
    jvp.dispose?.();
    throw new Error("a gradient JVP adapter requires a single-root tape");
  }
  if (jvp.kernel.numDirections !== diffVars.length) {
    jvp.dispose?.();
    throw new Error("gradient JVP direction count does not match diffVars");
  }
  const seeds = identitySeeds(tape, diffVars);
  return {
    varSlots: jvp.varSlots,
    numVars: jvp.numVars,
    backend: jvp.backend,
    kernel: {
      kind: "sync-grad",
      diffVars,
      eval(vars) {
        const result = jvp.kernel.evalPacked(packValues(tape, vars), seeds);
        return { val: result.vals[0]!, gradient: result.tangents[0]! };
      },
    },
    dispose: jvp.dispose,
  };
}

/** Adapt a seeded, multi-root synchronous JVP to an identity-seeded Jacobian. */
export function adaptSyncJvpToJacobian(
  tape: CompiledTape,
  diffVars: readonly VarName[],
  envelope: KernelEnvelope<JvpKernel>,
): KernelEnvelope<SyncJacobianKernel> {
  const jvp = requireSyncJvp(envelope);
  if (jvp.kernel.numDirections !== diffVars.length) {
    jvp.dispose?.();
    throw new Error("Jacobian JVP direction count does not match diffVars");
  }
  const seeds = identitySeeds(tape, diffVars);
  return {
    varSlots: jvp.varSlots,
    numVars: jvp.numVars,
    backend: jvp.backend,
    kernel: {
      kind: "sync-jacobian",
      numRoots: jvp.kernel.numRoots,
      diffVars,
      eval(vars) {
        const result = jvp.kernel.evalPacked(packValues(tape, vars), seeds);
        return { vals: result.vals, jacobian: result.tangents };
      },
    },
    dispose: jvp.dispose,
  };
}
