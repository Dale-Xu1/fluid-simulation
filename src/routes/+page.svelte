<script lang="ts">
import { onMount } from "svelte"

import Device, { ComputePass, ComputePipeline, LoadOperation, RenderPass, RenderPipeline, VertexFormat }
    from "../lib/Device"
import { Color4, Vector2 } from "../lib/Math"
import { Buffer, BufferFormat, Sampler, Shader, Texture, TextureFormat } from "../lib/Resource"

import shaderCode from "../lib/shaders/shader.wgsl?raw"
import computeCode from "../lib/shaders/compute.wgsl?raw"
import code from "../lib/shaders/test.wgsl?raw"

let canvas: HTMLCanvasElement
onMount(async () =>
{
    let ratio = window.devicePixelRatio

    canvas.width = window.innerWidth * ratio
    canvas.height = window.innerHeight * ratio

    let device = await Device.init(canvas)

    const HEIGHT = 200;
    const WIDTH = Math.floor(HEIGHT * canvas.width / canvas.height);
    let texture = new Texture(device, TextureFormat.RGBA_UNORM,
        GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST, [WIDTH, HEIGHT])

    let data = []
    for (let i = 0; i < WIDTH * HEIGHT * 4; i++) data[i] = (i + 1) % 4 === 0 ? 255 : Math.floor(Math.random() * 256)
    texture.write(new Uint8Array(data))

    let vertexBuffer = new Buffer(device, BufferFormat.F32, GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST, 2 * 6)
    vertexBuffer.write(
    [
        new Vector2(-1, -1), new Vector2( 1, -1), new Vector2( 1,  1),
        new Vector2(-1, -1), new Vector2( 1,  1), new Vector2(-1,  1)
    ])

    let uvBuffer = new Buffer(device, BufferFormat.F32, GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST, 2 * 6)
    uvBuffer.write(
    [
        new Vector2(0, 0), new Vector2(1, 0), new Vector2(1, 1),
        new Vector2(0, 0), new Vector2(1, 1), new Vector2(0, 1)
    ])

    let shader = new Shader(device, code)
    let pipeline = new RenderPipeline(device, shader, device.format,
        [{ format: VertexFormat.F32_2 }, { format: VertexFormat.F32_2 }])
    let pass = new RenderPass(pipeline, [[new Sampler(device), texture]], [vertexBuffer, uvBuffer])

    device.beginPass(device.texture, { load: LoadOperation.CLEAR, color: Color4.BLACK })
    pass.render(6)
    device.endPass()

    device.submit()


    // let device = await Device.init(canvas)

    // let vertexBuffer = new Buffer(device, BufferFormat.F32, GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST, 2 * 6)
    // vertexBuffer.write(
    // [
    //     new Vector2(-1, -1),
    //     new Vector2( 1, -1),
    //     new Vector2( 1,  1),
    //     new Vector2(-1, -1),
    //     new Vector2( 1,  1),
    //     new Vector2(-1,  1)
    // ])

    // const HEIGHT = 800;
    // const WIDTH = Math.floor(HEIGHT * canvas.width / canvas.height);

    // let uniformBuffer = new Buffer(device, BufferFormat.F32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 2)
    // uniformBuffer.write([WIDTH, HEIGHT])

    // let stateBuffers =
    // [
    //     new Buffer(device, BufferFormat.U32, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, WIDTH * HEIGHT),
    //     new Buffer(device, BufferFormat.U32, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, WIDTH * HEIGHT)
    // ]

    // let data = []
    // for (let i = 0; i < WIDTH * HEIGHT; i++) data[i] = Math.random() < 0.5 ? 0 : 1
    // stateBuffers[0].write(data)

    // let shader = new Shader(device, shaderCode)
    // let compute = new Shader(device, computeCode)

    // let renderPipeline = new RenderPipeline(device, shader, device.format, [{ format: VertexFormat.F32_2 }])
    // let renderPass1 = new RenderPass(renderPipeline, [[uniformBuffer, stateBuffers[1]]], [vertexBuffer])
    // let renderPass2 = new RenderPass(renderPipeline, [[uniformBuffer, stateBuffers[0]]], [vertexBuffer])

    // let computePipeline = new ComputePipeline(device, compute)
    // let computePass1 = new ComputePass(computePipeline, [[uniformBuffer, stateBuffers[0], stateBuffers[1]]])
    // let computePass2 = new ComputePass(computePipeline, [[uniformBuffer, stateBuffers[1], stateBuffers[0]]])

    // let i = 0
    // function loop()
    // {
    //     let renderPass = i % 2 === 0 ? renderPass1 : renderPass2
    //     let computePass = i % 2 === 0 ? computePass1 : computePass2
    //     i++

    //     computePass.dispatch(Math.ceil(WIDTH / 8), Math.ceil(HEIGHT / 8))

    //     device.beginPass(device.texture, { load: LoadOperation.CLEAR, color: Color4.BLACK })
    //     renderPass.render(6, WIDTH * HEIGHT)
    //     device.endPass()

    //     device.submit()
    //     requestAnimationFrame(loop)
    // }
    // loop()
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
