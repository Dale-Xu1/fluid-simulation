const DT: f32 = 1;
const SCALE: f32 = 1;

const VELOCITY_DIFFUSION: f32 = 0.99;
const WET_MASK_DIFFUSION: f32 = 0.95;
const DENSITY_DIFFUSION: f32 = 1;

const CURL: f32 = 0.01;
const STAINING: f32 = 5;

@group(0) @binding(0) var texture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<uniform> dimensions: vec2u;
@group(0) @binding(2) var<uniform> densityDimensions: vec2u;

@group(1) @binding(0) var<storage, read_write> wetMask: array<f32>;

@group(1) @binding(1) var<storage, read_write> velocity: array<vec2f>;
@group(1) @binding(2) var<storage, read_write> previousVelocity: array<vec2f>;

@group(1) @binding(3) var<storage, read_write> divergence: array<f32>;
@group(1) @binding(4) var<storage, read_write> curl: array<f32>;

@group(1) @binding(5) var<storage, read_write> pressure: array<f32>;
@group(1) @binding(6) var<storage, read_write> previousPressure: array<f32>;

@group(1) @binding(7) var<storage, read_write> density: array<vec3f>;
@group(1) @binding(8) var<storage, read_write> previousDensity: array<vec3f>;
@group(1) @binding(9) var<storage, read_write> pigment: array<vec3f>;

// Helper function to index buffer using two dimensional index
fn uindex(id: vec2u) -> u32 { return id.y * dimensions.x + id.x; }
fn indexf32(data: ptr<storage, array<f32>, read_write>, id: vec2i) -> f32
{
    let clamped = clamp(id, vec2i(0), vec2i(dimensions - 1));
    let i = uindex(vec2u(clamped));

    return data[i];
}

fn index2f(data: ptr<storage, array<vec2f>, read_write>, id: vec2i) -> vec2f
{
    if (id.x < 0 || id.x >= i32(dimensions.x) || id.y < 0 || id.y >= i32(dimensions.y)) { return vec2f(0); }

    let i = uindex(vec2u(id));
    return data[i];
}

// Bilinear interpolation functions
fn bilinear(data: ptr<storage, array<vec2f>, read_write>, id: vec2f) -> vec2f
{
    let i = vec2i(floor(id)); let ij = vec4i(i, i + 1);
    let t = id - vec2f(i);

    return mix(
        mix(index2f(data, ij.xy), index2f(data, ij.xw), t.y),
        mix(index2f(data, ij.zy), index2f(data, ij.zw), t.y), t.x);
}

@compute @workgroup_size(8, 8)
fn advection(@builtin(global_invocation_id) id: vec3u)
{
    if (id.x >= dimensions.x || id.y >= dimensions.y) { return; }

    // Move velocity vectors based on velocity field
    let i = uindex(id.xy);
    let previous = vec2f(id.xy) - previousVelocity[i] * DT;

    // Interpolation of four pixels surrounding previous position
    velocity[i] = VELOCITY_DIFFUSION * bilinear(&previousVelocity, previous);
}

@compute @workgroup_size(8, 8)
fn calculateDivergence(@builtin(global_invocation_id) id: vec3u)
{
    if (id.x >= dimensions.x || id.y >= dimensions.y) { return; }

    let i = vec2i(id.xy);
    let u = index2f(&velocity, i + vec2i(0, 1)).y;
    let d = index2f(&velocity, i - vec2i(0, 1)).y;
    let r = index2f(&velocity, i + vec2i(1, 0)).x;
    let l = index2f(&velocity, i - vec2i(1, 0)).x;

    divergence[uindex(id.xy)] = 0.5 * (u - d + r - l);
}

@compute @workgroup_size(8, 8)
fn calculateCurl(@builtin(global_invocation_id) id: vec3u)
{
    if (id.x >= dimensions.x || id.y >= dimensions.y) { return; }

    let i = vec2i(id.xy);
    let u = index2f(&velocity, i + vec2i(0, 1)).x;
    let d = index2f(&velocity, i - vec2i(0, 1)).x;
    let r = index2f(&velocity, i + vec2i(1, 0)).y;
    let l = index2f(&velocity, i - vec2i(1, 0)).y;

    curl[uindex(id.xy)] = d - u + l - r;
}

@compute @workgroup_size(8, 8)
fn vorticity(@builtin(global_invocation_id) id: vec3u)
{
    if (id.x >= dimensions.x || id.y >= dimensions.y) { return; }

    let position = vec2i(id.xy);
    let i = uindex(id.xy);

    let m = curl[i];
    let u = indexf32(&curl, position + vec2i(0, 1));
    let d = indexf32(&curl, position - vec2i(0, 1));
    let r = indexf32(&curl, position + vec2i(1, 0));
    let l = indexf32(&curl, position - vec2i(1, 0));

    let force = vec2f(abs(u) - abs(d), abs(r) - abs(l));
    let ls = max(0.001, dot(force, force));

    velocity[i] += CURL * m * force * inverseSqrt(ls) * DT;
}

@compute @workgroup_size(8, 8)
fn jacobi(@builtin(global_invocation_id) id: vec3u)
{
    if (id.x >= dimensions.x || id.y >= dimensions.y) { return; }

    let position = vec2i(id.xy);
    let i = uindex(id.xy);

    let u = indexf32(&previousPressure, position + vec2i(0, 1));
    let d = indexf32(&previousPressure, position - vec2i(0, 1));
    let r = indexf32(&previousPressure, position + vec2i(1, 0));
    let l = indexf32(&previousPressure, position - vec2i(1, 0));
    let div = divergence[i];

    pressure[i] = 0.25 * (u + d + r + l - div);
}

@compute @workgroup_size(8, 8)
fn gradientSubtraction(@builtin(global_invocation_id) id: vec3u)
{
    if (id.x >= dimensions.x || id.y >= dimensions.y) { return; }

    let position = vec2i(id.xy);
    let u = indexf32(&pressure, position + vec2i(0, 1));
    let d = indexf32(&pressure, position - vec2i(0, 1));
    let r = indexf32(&pressure, position + vec2i(1, 0));
    let l = indexf32(&pressure, position - vec2i(1, 0));

    velocity[uindex(id.xy)] -= 0.5 * vec2f(r - l, u - d) * DT;
}

@compute @workgroup_size(8, 8)
fn wetness(@builtin(global_invocation_id) id: vec3u)
{
    if (id.x >= dimensions.x || id.y >= dimensions.y) { return; }

    let i = uindex(id.xy);
    // let position = vec2i(id.xy);

    // let m = wetMask[i];
    // let u = indexf32(&wetMask, position + vec2i(0, 1));
    // let d = indexf32(&wetMask, position - vec2i(0, 1));
    // let r = indexf32(&wetMask, position + vec2i(1, 0));
    // let l = indexf32(&wetMask, position - vec2i(1, 0));

    // wetMask[i] = WET_MASK_DIFFUSION * 0.2 * (m + u + d + r + l);
    // wetMask[i] *= WET_MASK_DIFFUSION;
    if (wetMask[i] < 0.1) { velocity[i] = vec2f(0); }
}

// Helper function to index buffer using two dimensional index
fn sindex(id: vec2u) -> u32 { return id.y * densityDimensions.x + id.x; }
fn index3f(data: ptr<storage, array<vec3f>, read_write>, id: vec2i) -> vec3f
{
    if (id.x < 0 || id.x >= i32(densityDimensions.x) || id.y < 0 || id.y >= i32(densityDimensions.y)) { return vec3f(0); }

    let i = sindex(vec2u(id));
    return data[i];
}

fn bilinear3f(data: ptr<storage, array<vec3f>, read_write>, id: vec2f) -> vec3f
{
    let i = vec2i(floor(id)); let ij = vec4i(i, i + 1);
    let t = id - vec2f(i);

    return mix(
        mix(index3f(data, ij.xy), index3f(data, ij.xw), t.y),
        mix(index3f(data, ij.zy), index3f(data, ij.zw), t.y), t.x);
}

@compute @workgroup_size(8, 8)
fn densityAdvection(@builtin(global_invocation_id) id: vec3u)
{
    if (id.x >= densityDimensions.x || id.y >= densityDimensions.y) { return; }

    // Move density based on velocity field
    let i = vec2f(id.xy);
    let previous = vec2f(id.xy) - SCALE * bilinear(&velocity, i / SCALE) * DT;

    density[sindex(id.xy)] = DENSITY_DIFFUSION * bilinear3f(&previousDensity, previous);
}

@compute @workgroup_size(8, 8)
fn transferPigment(@builtin(global_invocation_id) id: vec3u)
{
    if (id.x >= densityDimensions.x || id.y >= densityDimensions.y) { return; }

    let i = sindex(id.xy);
    let down = density[i];
    let up = pigment[i] / STAINING;

    pigment[i] += down - up;
    density[i] += up - down;
}

@compute @workgroup_size(8, 8)
fn transfer(@builtin(global_invocation_id) id: vec3u)
{
    // Transfer density data as colors to texture
    let i = sindex(id.xy);
    let color = vec4f(density[i] + pigment[i], 1);

    textureStore(texture, id.xy, color);
}

// User input passes
@group(2) @binding(0) var<uniform> position: vec2f;
@group(2) @binding(1) var<uniform> impulse: vec2f;
@group(2) @binding(2) var<uniform> color: vec3f;
@group(2) @binding(3) var<uniform> radius: f32;

@compute @workgroup_size(8, 8)
fn addImpulse(@builtin(global_invocation_id) id: vec3u)
{
    let offset = vec2f(id.xy) - vec2f(radius);
    if (dot(offset, offset) > radius * radius)  { return; }

    // Ensure index is within boundaries
    let ij = vec2i(position + offset);
    if (ij.x < 0 || ij.x >= i32(dimensions.x) || ij.y < 0 || ij.y >= i32(dimensions.y)) { return; }

    let i = uindex(vec2u(ij));
    wetMask[i] = 1;
    velocity[i] += impulse;
}

@compute @workgroup_size(8, 8)
fn addDensity(@builtin(global_invocation_id) id: vec3u)
{
    let offset = vec2f(id.xy) - vec2f(radius);
    if (dot(offset, offset) > radius * radius)  { return; }

    // Ensure index is within boundaries
    let ij = vec2i(SCALE * position + offset);
    if (ij.x < 0 || ij.x >= i32(densityDimensions.x) || ij.y < 0 || ij.y >= i32(densityDimensions.y)) { return; }

    density[sindex(vec2u(ij))] += color;
}
