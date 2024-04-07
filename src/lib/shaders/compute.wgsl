@group(0) @binding(0) var<uniform> grid: vec2f;
@group(0) @binding(1) var<storage> stateIn: array<u32>;
@group(0) @binding(2) var<storage, read_write> stateOut: array<u32>;

fn index(id: vec2u) -> u32 { return (id.y % u32(grid.y)) * u32(grid.x) + (id.x % u32(grid.x)); }

@compute
@workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u)
{
    let i = index(id.xy);
    let neighbors = stateIn[index(vec2u(id.x + 1, id.y + 1))]
                  + stateIn[index(vec2u(id.x + 1, id.y    ))]
                  + stateIn[index(vec2u(id.x + 1, id.y - 1))]
                  + stateIn[index(vec2u(id.x    , id.y - 1))]
                  + stateIn[index(vec2u(id.x - 1, id.y - 1))]
                  + stateIn[index(vec2u(id.x - 1, id.y    ))]
                  + stateIn[index(vec2u(id.x - 1, id.y + 1))]
                  + stateIn[index(vec2u(id.x    , id.y + 1))];


    if (neighbors == 2) { stateOut[i] = stateIn[i]; }
    else if (neighbors == 3) { stateOut[i] = 1; }
    else { stateOut[i] = 0; }
}
