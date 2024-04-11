<script lang="ts">
import { onMount } from "svelte"

import Device, { ComputePass, ComputePipeline, LoadOperation, RenderPass, RenderPipeline, VertexFormat } from "../lib/Device"
import { Color4, Vector2 } from "../lib/Math"
import { Buffer, BufferFormat, Sampler, Shader, Texture, TextureFormat } from "../lib/Resource"

import textureCode from "../lib/shaders/texture.wgsl?raw"
import fluidSimulationCode from "../lib/shaders/fluid-simulation.wgsl?raw"

let canvas: HTMLCanvasElement
onMount(async () =>
{
    let ratio = window.devicePixelRatio

    canvas.width = window.innerWidth * ratio
    canvas.height = window.innerHeight * ratio

    let device = await Device.init(canvas)

    const HEIGHT = 200;
    const WIDTH = Math.floor(HEIGHT * canvas.width / canvas.height);

    let texture = new Texture(device, TextureFormat.RGBA_UNORM, GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST, [WIDTH, HEIGHT])
    let dimensionsBuffer = new Buffer(device, BufferFormat.U32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 2)
    dimensionsBuffer.write([WIDTH, HEIGHT])

    let vertexBuffer = new Buffer(device, BufferFormat.F32, GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST, 2 * 6)
    vertexBuffer.write(
    [
        new Vector2(-1, -1), new Vector2(1, -1), new Vector2( 1, 1),
        new Vector2(-1, -1), new Vector2(1,  1), new Vector2(-1, 1)
    ])

    let pipeline = new RenderPipeline(device, new Shader(device, textureCode), device.format, [{ format: VertexFormat.F32_2 }])
    let pass = new RenderPass(pipeline, [[{ i: 0, resource: new Sampler(device) }, { i: 1, resource: texture }]], [vertexBuffer])


    let velocityBuffer = new Buffer(device, BufferFormat.F32, GPUBufferUsage.STORAGE, WIDTH * HEIGHT * 2)
    let previousVelocityBuffer = new Buffer(device, BufferFormat.F32, GPUBufferUsage.STORAGE, WIDTH * HEIGHT * 2)

    let divergenceBuffer = new Buffer(device, BufferFormat.F32, GPUBufferUsage.STORAGE, WIDTH * HEIGHT)
    let pressureBuffer = new Buffer(device, BufferFormat.F32, GPUBufferUsage.STORAGE, WIDTH * HEIGHT)
    let previousPressureBuffer = new Buffer(device, BufferFormat.F32, GPUBufferUsage.STORAGE, WIDTH * HEIGHT)

    let densityBuffer = new Buffer(device, BufferFormat.F32, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, WIDTH * HEIGHT * 4)
    let previousDensityBuffer = new Buffer(device, BufferFormat.F32, GPUBufferUsage.STORAGE, WIDTH * HEIGHT * 4)

    let data = []
    for (let i = 0; i < WIDTH * HEIGHT; i++) data[i] = new Color4(Math.random(), Math.random(), Math.random(), 1)
    densityBuffer.write(data)

    let advectionPass = new ComputePass(new ComputePipeline(device, new Shader(device, fluidSimulationCode), "advection"),
        [[{ i: 1, resource: dimensionsBuffer }, { i: 2, resource: velocityBuffer }, { i: 3, resource: previousVelocityBuffer }]])
    let transferPass = new ComputePass(new ComputePipeline(device, new Shader(device, fluidSimulationCode), "transfer"),
        [[{ i: 0, resource: texture }, { i: 1, resource: dimensionsBuffer }, { i: 7, resource: densityBuffer }]])

    loop()
    function loop()
    {
        let w = Math.ceil(WIDTH / 8), h = Math.ceil(HEIGHT / 8)

        advectionPass.dispatch(w, h)
        transferPass.dispatch(w, h)

        device.beginPass(device.texture, { load: LoadOperation.CLEAR, color: Color4.BLACK })
        pass.render(6)
        device.endPass()

        device.submit()
        requestAnimationFrame(loop)
    }
})

</script>

<canvas bind:this={canvas}></canvas>
<style>
:global(*) {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

:global(body) {
    overflow: hidden;
}

canvas {
    width: 100%;
    height: 100vh;
}

</style>
