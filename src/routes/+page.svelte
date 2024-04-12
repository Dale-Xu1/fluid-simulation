<script lang="ts">
import { onMount } from "svelte"

import Device, { ComputePass, ComputePipeline, LoadOperation, RenderPass, RenderPipeline, VertexFormat } from "../lib/Device"
import { Color3, Color4, Vector2 } from "../lib/Math"
import { Buffer, BufferFormat, Sampler, Shader, Texture, TextureFormat } from "../lib/Resource"
import Input, { MouseButton } from "../lib/Input"

import textureCode from "../lib/shaders/texture.wgsl?raw"
import fluidSimulationCode from "../lib/shaders/fluid-simulation.wgsl?raw"

let canvas: HTMLCanvasElement
onMount(() =>
{
    let ratio = window.devicePixelRatio

    canvas.width = window.innerWidth * ratio
    canvas.height = window.innerHeight * ratio
    main()
})

async function main()
{
    let device = await Device.init(canvas)

    const ITERATIONS = 32

    const HEIGHT = 300
    const WIDTH = Math.floor(HEIGHT * canvas.width / canvas.height)

    const RADIUS = 10

    // Initialize rendering texture related data
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

    // Initialize fluid simulation buffers
    let velocityBuffer = new Buffer(device, BufferFormat.F32, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, WIDTH * HEIGHT * 2)
    let previousVelocityBuffer = new Buffer(device, BufferFormat.F32, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, WIDTH * HEIGHT * 2)

    let divergenceBuffer = new Buffer(device, BufferFormat.F32, GPUBufferUsage.STORAGE, WIDTH * HEIGHT)
    let curlBuffer = new Buffer(device, BufferFormat.F32, GPUBufferUsage.STORAGE, WIDTH * HEIGHT)

    let pressureBuffer = new Buffer(device, BufferFormat.F32, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, WIDTH * HEIGHT)
    let previousPressureBuffer = new Buffer(device, BufferFormat.F32, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, WIDTH * HEIGHT)

    let densityBuffer = new Buffer(device, BufferFormat.F32, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, WIDTH * HEIGHT * 4)
    let previousDensityBuffer = new Buffer(device, BufferFormat.F32, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, WIDTH * HEIGHT * 4)

    let shader = new Shader(device, fluidSimulationCode)

    let advectionPass = new ComputePass(new ComputePipeline(device, shader, "advection"),
        [[{ i: 1, resource: dimensionsBuffer }, { i: 2, resource: velocityBuffer }, { i: 3, resource: previousVelocityBuffer }]])
    let calculateDivergencePass = new ComputePass(new ComputePipeline(device, shader, "calculateDivergence"),
        [[{ i: 1, resource: dimensionsBuffer }, { i: 2, resource: velocityBuffer }, { i: 4, resource: divergenceBuffer }]])
    let calculateCurlPass = new ComputePass(new ComputePipeline(device, shader, "calculateCurl"),
        [[{ i: 1, resource: dimensionsBuffer }, { i: 2, resource: velocityBuffer }, { i: 5, resource: curlBuffer }]])
    let vorticityPass = new ComputePass(new ComputePipeline(device, shader, "vorticity"),
        [[{ i: 1, resource: dimensionsBuffer }, { i: 2, resource: velocityBuffer }, { i: 5, resource: curlBuffer }]])
    let jacobiPass = new ComputePass(new ComputePipeline(device, shader, "jacobi"),
        [[{ i: 1, resource: dimensionsBuffer }, { i: 4, resource: divergenceBuffer }, { i: 6, resource: pressureBuffer }, { i: 7, resource: previousPressureBuffer }]])
    let gradientSubtractionPass = new ComputePass(new ComputePipeline(device, shader, "gradientSubtraction"),
        [[{ i: 1, resource: dimensionsBuffer }, { i: 2, resource: velocityBuffer }, { i: 6, resource: pressureBuffer }]])
    let densityAdvectionPass = new ComputePass(new ComputePipeline(device, shader, "densityAdvection"),
        [[{ i: 1, resource: dimensionsBuffer }, { i: 3, resource: previousVelocityBuffer }, { i: 8, resource: densityBuffer }, { i: 9, resource: previousDensityBuffer }]])

    let transferPass = new ComputePass(new ComputePipeline(device, shader, "transfer"),
        [[{ i: 0, resource: texture }, { i: 1, resource: dimensionsBuffer }, { i: 8, resource: densityBuffer }]])

    // Initialize user interface uniforms
    let positionBuffer = new Buffer(device, BufferFormat.F32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 2)
    let impulseBuffer = new Buffer(device, BufferFormat.F32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 2)
    let colorBuffer = new Buffer(device, BufferFormat.F32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 3)

    let addDensityPass = new ComputePass(new ComputePipeline(device, shader, "addDensity"),
        [[{ i: 1, resource: dimensionsBuffer }, { i: 2, resource: velocityBuffer }, { i: 8, resource: densityBuffer }], [{ i: 0, resource: positionBuffer }, { i: 1, resource: impulseBuffer }, { i: 2, resource: colorBuffer }]])

    let w = Math.ceil(WIDTH / 8), h = Math.ceil(HEIGHT / 8)
    let r = Math.ceil((2 * RADIUS + 1) / 8)

    let previous = Input.mouse

    loop()
    function loop()
    {
        let position = Input.mouse
        if (Input.button[MouseButton.LEFT])
        {
            let localPosition = new Vector2(position.x * WIDTH / canvas.width, HEIGHT - position.y * HEIGHT / canvas.height)
            let localPrevious = new Vector2(previous.x * WIDTH / canvas.width, HEIGHT - previous.y * HEIGHT / canvas.height)
            let impulse = localPosition.sub(localPrevious)

            positionBuffer.write([localPosition])
            impulseBuffer.write([impulse])
            colorBuffer.write([Color3.WHITE.mul(0.15 * Math.sqrt(impulse.length))])

            addDensityPass.dispatch(r, r)
        }
        previous = position

        device.copyBuffer(velocityBuffer, previousVelocityBuffer)
        device.copyBuffer(densityBuffer, previousDensityBuffer)

        advectionPass.dispatch(w, h)
        calculateDivergencePass.dispatch(w, h)
        calculateCurlPass.dispatch(w, h)
        vorticityPass.dispatch(w, h)

        for (let i = 0; i < ITERATIONS; i++)
        {
            device.copyBuffer(pressureBuffer, previousPressureBuffer)
            jacobiPass.dispatch(w, h)
        }

        gradientSubtractionPass.dispatch(w, h)
        densityAdvectionPass.dispatch(w, h)

        transferPass.dispatch(w, h)

        device.beginPass(device.texture, { load: LoadOperation.CLEAR, color: Color4.BLACK })
        pass.render(6)
        device.endPass()

        device.submit()
        requestAnimationFrame(loop)
    }
}

</script>

<svelte:head>
    <title>Fluid Simulation</title>
</svelte:head>
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
