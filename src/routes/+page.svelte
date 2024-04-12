<script lang="ts">
import { onMount } from "svelte"

import Device, { ComputePass, ComputePipeline, LoadOperation, RenderPass, RenderPipeline, VertexFormat } from "../lib/Device"
import { Color3, Color4, Vector2 } from "../lib/Math"
import { Buffer, BufferFormat, Sampler, SamplerFilterMode, Shader, Texture, TextureFormat } from "../lib/Resource"
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

    const SCALE = 3
    const ITERATIONS = 32

    const RADIUS = 10

    let width = window.innerWidth, height = window.innerHeight
    let w = Math.floor(width / SCALE), h = Math.floor(height / SCALE)

    // Initialize rendering texture related data
    let texture = new Texture(device, TextureFormat.RGBA_UNORM, GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST, [width, height])
    let dimensionsBuffer = new Buffer(device, BufferFormat.U32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 2)
    let densityDimensionsBuffer = new Buffer(device, BufferFormat.U32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 2)
    dimensionsBuffer.write([w, h])
    densityDimensionsBuffer.write([width, height])

    let vertexBuffer = new Buffer(device, BufferFormat.F32, GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST, 2 * 6)
    vertexBuffer.write(
    [
        new Vector2(-1, -1), new Vector2(1, -1), new Vector2( 1, 1),
        new Vector2(-1, -1), new Vector2(1,  1), new Vector2(-1, 1)
    ])

    let pipeline = new RenderPipeline(device, new Shader(device, textureCode), device.format, [{ format: VertexFormat.F32_2 }])
    let pass = new RenderPass(pipeline, [[{ i: 0, resource: new Sampler(device, { mag: SamplerFilterMode.LINEAR }) }, { i: 1, resource: texture }]], [vertexBuffer])

    // Initialize fluid simulation buffers
    let velocityBuffer = new Buffer(device, BufferFormat.F32, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, w * h * 2)
    let previousVelocityBuffer = new Buffer(device, BufferFormat.F32, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, w * h * 2)

    let divergenceBuffer = new Buffer(device, BufferFormat.F32, GPUBufferUsage.STORAGE, w * h)
    let curlBuffer = new Buffer(device, BufferFormat.F32, GPUBufferUsage.STORAGE, w * h)

    let pressureBuffer = new Buffer(device, BufferFormat.F32, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, w * h)
    let previousPressureBuffer = new Buffer(device, BufferFormat.F32, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, w * h)

    let densityBuffer = new Buffer(device, BufferFormat.F32, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, width * height * 4)
    let previousDensityBuffer = new Buffer(device, BufferFormat.F32, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, width * height * 4)

    // Initialize shader passes
    let shader = new Shader(device, fluidSimulationCode)

    let advectionPass = new ComputePass(new ComputePipeline(device, shader, "advection"),
        [[{ i: 1, resource: dimensionsBuffer }], [{ i: 0, resource: velocityBuffer }, { i: 1, resource: previousVelocityBuffer }]])

    let calculateDivergencePass = new ComputePass(new ComputePipeline(device, shader, "calculateDivergence"),
        [[{ i: 1, resource: dimensionsBuffer }], [{ i: 0, resource: velocityBuffer }, { i: 2, resource: divergenceBuffer }]])
    let calculateCurlPass = new ComputePass(new ComputePipeline(device, shader, "calculateCurl"),
        [[{ i: 1, resource: dimensionsBuffer }], [{ i: 0, resource: velocityBuffer }, { i: 3, resource: curlBuffer }]])
    let vorticityPass = new ComputePass(new ComputePipeline(device, shader, "vorticity"),
        [[{ i: 1, resource: dimensionsBuffer }], [{ i: 0, resource: velocityBuffer }, { i: 3, resource: curlBuffer }]])

    let jacobiPass = new ComputePass(new ComputePipeline(device, shader, "jacobi"),
        [[{ i: 1, resource: dimensionsBuffer }], [{ i: 2, resource: divergenceBuffer }, { i: 4, resource: pressureBuffer }, { i: 5, resource: previousPressureBuffer }]])
    let gradientSubtractionPass = new ComputePass(new ComputePipeline(device, shader, "gradientSubtraction"),
        [[{ i: 1, resource: dimensionsBuffer }], [{ i: 0, resource: velocityBuffer }, { i: 4, resource: pressureBuffer }]])

    let densityAdvectionPass = new ComputePass(new ComputePipeline(device, shader, "densityAdvection"),
        [[{ i: 1, resource: dimensionsBuffer }, { i: 2, resource: densityDimensionsBuffer }], [{ i: 1, resource: previousVelocityBuffer }, { i: 6, resource: densityBuffer }, { i: 7, resource: previousDensityBuffer }]])
    let transferPass = new ComputePass(new ComputePipeline(device, shader, "transfer"),
        [[{ i: 0, resource: texture }, { i: 2, resource: densityDimensionsBuffer }], [{ i: 6, resource: densityBuffer }]])

    // Initialize user interface uniforms
    let positionBuffer = new Buffer(device, BufferFormat.F32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 2)
    let impulseBuffer = new Buffer(device, BufferFormat.F32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 2)
    let colorBuffer = new Buffer(device, BufferFormat.F32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 3)
    let radiusBuffer = new Buffer(device, BufferFormat.F32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 1)

    let addImpulsePass = new ComputePass(new ComputePipeline(device, shader, "addImpulse"),
        [[{ i: 1, resource: dimensionsBuffer }], [{ i: 0, resource: velocityBuffer }], [{ i: 0, resource: positionBuffer }, { i: 1, resource: impulseBuffer }, { i: 3, resource: radiusBuffer }]])
    let addDensityPass = new ComputePass(new ComputePipeline(device, shader, "addDensity"),
        [[{ i: 2, resource: densityDimensionsBuffer }], [{ i: 6, resource: densityBuffer }], [{ i: 0, resource: positionBuffer }, { i: 2, resource: colorBuffer }, { i: 3, resource: radiusBuffer }]])


    let x = Math.ceil(w / 8), y = Math.ceil(h / 8)
    let nx = Math.ceil(width / 8), ny = Math.ceil(height / 8)

    let r = Math.ceil((2 * RADIUS - 1) / 8)
    let nr = Math.ceil((2 * SCALE * RADIUS + 1) / 8)

    let previous = Input.mouse

    loop()
    function loop()
    {
        let position = Input.mouse
        if (Input.button[MouseButton.LEFT])
        {
            let localPosition = new Vector2(position.x * w / canvas.width, h - position.y * h / canvas.height)
            let localPrevious = new Vector2(previous.x * w / canvas.width, h - previous.y * h / canvas.height)
            let impulse = localPosition.sub(localPrevious)

            positionBuffer.write([localPosition])
            impulseBuffer.write([impulse])
            radiusBuffer.write([RADIUS - 1])
            addImpulsePass.dispatch(r, r)

            device.submit()
            colorBuffer.write([Color3.WHITE.mul(0.1 * Math.sqrt(impulse.length))])
            radiusBuffer.write([SCALE * RADIUS])
            addDensityPass.dispatch(nr, nr)
        }
        previous = position

        device.copyBuffer(velocityBuffer, previousVelocityBuffer)
        device.copyBuffer(densityBuffer, previousDensityBuffer)

        advectionPass.dispatch(x, y)
        calculateDivergencePass.dispatch(x, y)
        calculateCurlPass.dispatch(x, y)
        vorticityPass.dispatch(x, y)

        for (let i = 0; i < ITERATIONS; i++)
        {
            device.copyBuffer(pressureBuffer, previousPressureBuffer)
            jacobiPass.dispatch(x, y)
        }
        gradientSubtractionPass.dispatch(x, y)

        densityAdvectionPass.dispatch(nx, ny)
        transferPass.dispatch(nx, ny)

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
