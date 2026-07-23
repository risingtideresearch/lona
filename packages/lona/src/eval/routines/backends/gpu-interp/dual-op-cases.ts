/**
 * Shared non-leaf WGSL switch cases for GPU interpreter forward-mode AD.
 * The surrounding shader owns OP_LIT/OP_VAR because their storage layouts differ.
 */
export const GPU_INTERP_DUAL_OP_CASES = /* wgsl */ `
      case 10u { r = sqrt(va); fa = select(0.5 / r, 0.0, r == 0.0); }
      case 11u { r = cbrt_f32(va); fa = select(1.0 / (3.0 * r * r), 0.0, r == 0.0); }
      case 12u { r = cos(va); fa = -sin(va); }
      case 13u { r = acos(va); fa = -1.0 / sqrt(1.0 - va * va); }
      case 14u { r = asin(va); fa = 1.0 / sqrt(1.0 - va * va); }
      case 15u { r = tan(va); let c = cos(va); fa = 1.0 / (c * c); }
      case 16u { r = atan(va); fa = 1.0 / (1.0 + va * va); }
      case 17u { r = log(va); fa = 1.0 / va; }
      case 18u { r = exp(va); fa = r; }
      case 19u { r = abs(va); fa = select(select(-1.0, 0.0, va == 0.0), 1.0, va > 0.0); }
      case 20u { r = -va; fa = -1.0; }
      case 21u { r = sin(va); fa = cos(va); }
      case 22u { r = sign(va); }
      case 23u { r = select(0.0, 1.0, va == 0.0); }
      case 24u { r = tanh(va); fa = 1.0 - r * r; }
      case 25u { r = log(1.0 + va); fa = 1.0 / (1.0 + va); }
      case 26u { r = va; fa = 1.0; }
      case 40u { r = va + vb; fa = 1.0; fb = 1.0; }
      case 41u { r = va - vb; fa = 1.0; fb = -1.0; }
      case 42u { r = va * vb; fa = vb; fb = va; }
      case 43u {
        if (vb == 0.0) { r = DIV_ZERO; }
        else { r = va / vb; fa = 1.0 / vb; fb = -va / (vb * vb); }
      }
      case 44u { r = va % vb; fa = 1.0; }
      case 45u {
        r = atan2(va, vb);
        let denom = va * va + vb * vb;
        if (denom != 0.0) { fa = vb / denom; fb = -va / denom; }
      }
      case 46u { let useA = va <= vb; r = select(vb, va, useA); fa = select(0.0, 1.0, useA); fb = select(1.0, 0.0, useA); }
      case 47u { let useA = va >= vb; r = select(vb, va, useA); fa = select(0.0, 1.0, useA); fb = select(1.0, 0.0, useA); }
      case 48u { r = sign(va - vb); }
      case 49u { let z = va == 0.0; r = select(va, vb, z); fa = select(1.0, 0.0, z); fb = select(0.0, 1.0, z); }
      case 50u { let nz = va != 0.0; r = select(va, vb, nz); fa = select(1.0, 0.0, nz); fb = select(0.0, 1.0, nz); }
      case 51u { r = va; fa = 1.0; }
      case 52u { r = va; fa = 1.0; }
`;
