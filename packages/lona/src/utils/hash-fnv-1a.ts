const FNV_OFFSET_BASIS = 2166136261;
const FNV_PRIME = 16777619;
const DEFAULT_SEED = FNV_OFFSET_BASIS;
const HASH_CACHE_LIMIT = 4096;
const encoder = new TextEncoder();
const stringHashCache = new Map<string, number>();
const numberHashCache = new Map<number, number>();

function rememberHash<K>(cache: Map<K, number>, key: K, hash: number): void {
  if (cache.size >= HASH_CACHE_LIMIT) {
    cache.clear();
  }
  cache.set(key, hash);
}

function hashUtf8String(value: string, seed: number): number {
  let hash = seed;
  let isAscii = true;

  for (let i = 0; i < value.length; i++) {
    const code = value.charCodeAt(i);
    if (code > 0x7f) {
      isAscii = false;
      break;
    }
    hash ^= code;
    hash = Math.imul(hash, FNV_PRIME);
  }

  if (isAscii) {
    return hash >>> 0;
  }

  hash = seed;
  const bytes = encoder.encode(value);
  for (let i = 0; i < bytes.length; i++) {
    hash ^= bytes[i]!;
    hash = Math.imul(hash, FNV_PRIME);
  }
  return hash >>> 0;
}

export function hashValue(
  value: string | number,
  seed: number = DEFAULT_SEED,
): number {
  if (typeof value !== "string" && typeof value !== "number") {
    throw new TypeError("Input must be a string or number");
  }

  if (typeof value === "number") {
    if (!Number.isFinite(value)) {
      return Number.isNaN(value) ? 0x01234567 : 0x89abcdef;
    }

    let normalized = value;
    if (Math.floor(normalized) !== normalized) {
      normalized = Number(normalized.toFixed(10));
    }

    if (seed === DEFAULT_SEED) {
      const cached = numberHashCache.get(normalized);
      if (cached !== undefined) {
        return cached;
      }
    }

    const hash = hashUtf8String(String(normalized), seed);
    if (seed === DEFAULT_SEED) {
      rememberHash(numberHashCache, normalized, hash);
    }
    return hash;
  }

  if (seed === DEFAULT_SEED) {
    const cached = stringHashCache.get(value);
    if (cached !== undefined) {
      return cached;
    }
  }

  const hash = hashUtf8String(value, seed);
  if (seed === DEFAULT_SEED) {
    rememberHash(stringHashCache, value, hash);
  }
  return hash;
}

export function combineHashes(
  valueHash: number,
  childrenHashes?: number[],
): number {
  // Start with the node's own value hash
  let result = valueHash & 0xffffffff;

  // Handle undefined or null childrenHashes
  if (!childrenHashes || childrenHashes.length === 0) {
    return result >>> 0;
  }

  // Add length information to make different sized arrays hash differently
  result = mixHash(result ^ (childrenHashes.length & 0xff));

  // Incorporate each child's hash with proper mixing and position awareness
  for (let i = 0; i < childrenHashes.length; i++) {
    const childHash = childrenHashes[i] || 0; // Default to 0 if child hash is undefined

    // Mix position information to ensure order matters
    // Each child contributes differently based on its position in the array
    const positionedChildHash = childHash ^ Math.imul(i + 1, 0x9e3779b9); // Prime constant

    // Mix in the child hash
    result ^= positionedChildHash;
    result = mixHash(result);
  }

  // Final avalanche mix for better distribution
  return finalizeHash(result);
}

// Helper function for hash mixing (based on MurmurHash3)
function mixHash(hash: number): number {
  hash = Math.imul(hash, 0x85ebca6b);
  // Correct bit rotation (right rotate by 13)
  hash = ((hash >>> 13) | (hash << 19)) >>> 0;
  hash = Math.imul(hash, 0xc2b2ae35);
  return hash >>> 0;
}

// Finalization mix - force all bits of the hash to avalanche
function finalizeHash(hash: number): number {
  hash ^= hash >>> 16;
  hash = Math.imul(hash, 0x85ebca6b);
  hash ^= hash >>> 13;
  hash = Math.imul(hash, 0xc2b2ae35);
  hash ^= hash >>> 16;
  return hash >>> 0; // Ensure unsigned 32-bit integer
}
