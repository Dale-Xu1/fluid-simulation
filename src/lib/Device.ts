import { Color4 } from "./Math"
import type Resource from "./Resource"
import { Buffer, Shader, Texture, TextureFormat } from "./Resource"

export const enum StepMode { VERTEX = "vertex", INSTANCE = "instance" }
export const enum VertexFormat
{
    I32 = "sint32",  I32_2 = "sint32x2",  I32_3 = "sint32x3",  I32_4 = "sint32x4",
    U32 = "uint32",  U32_2 = "uint32x2",  U32_3 = "uint32x3",  U32_4 = "uint32x4",
    F32 = "float32", F32_2 = "float32x2", F32_3 = "float32x3", F32_4 = "float32x4"
}

export const enum LoadOperation { CLEAR = "clear", LOAD = "load" }

export const enum Primitive { POINT = "point-list", LINE = "line-list", TRIANGLE = "triangle-list" }
export const enum CullMode { NONE = "none", FRONT = "front", BACK = "back" }

export interface RenderEncoderParams
{

    load?: LoadOperation
    color?: Color4

    depth?: Texture
    depthLoad?: LoadOperation

}

export default class Device
{

    public static async init(canvas: HTMLCanvasElement): Promise<Device>
    {
        // Request device and context
        let adapter = await window.navigator.gpu.requestAdapter()
        if (adapter === null) throw new Error("No GPU adapter found")

        let device = await adapter.requestDevice()

        let context = canvas.getContext("webgpu")
        if (context === null) throw new Error("No WebGPU context found")

        // Configure texture format for canvas context
        let format = window.navigator.gpu.getPreferredCanvasFormat()
        context.configure({ device, format })

        return new Device(device, context, format as TextureFormat)
    }

    private target!: Texture
    public get texture(): Texture
    {
        // Maintain texture reference (create new object if canvas texture changes)
        let current = this.context.getCurrentTexture()
        if (!this.target || current !== this.target.texture) this.target = new Texture(this, current)

        return this.target
    }

    public encoder: GPUCommandEncoder
    public renderEncoder!: GPURenderPassEncoder

    private constructor(public readonly device: GPUDevice, private readonly context: GPUCanvasContext,
        public readonly format: TextureFormat)
    {
        this.encoder = this.device.createCommandEncoder()
    }


    public beginPass(texture: Texture,
    {
        load = LoadOperation.CLEAR, color = Color4.WHITE,
        depth, depthLoad = load
    }: RenderEncoderParams = {})
    {
        // Begin render pass
        this.renderEncoder = this.encoder.beginRenderPass(
        {
            colorAttachments:
            [{
                view: texture.view,
                clearValue: color, loadOp: load, storeOp: "store"
            }],
            depthStencilAttachment: depth ?
            {
                view: depth.view,
                depthClearValue: 1, depthLoadOp: depthLoad, depthStoreOp: "store"
            } : undefined
        })
    }

    public endPass() { this.renderEncoder.end() }
    public submit()
    {
        // Submit encoded instructions to GPU and reset encoder
        this.device.queue.submit([this.encoder.finish()])
        this.encoder = this.device.createCommandEncoder()
    }

    public copyBuffer(source: Buffer, destination: Buffer, sourceOffset: number = 0, destinationOffset: number = 0,
        length?: number)
    {
        length ??= source.length
        this.encoder.copyBufferToBuffer(source.buffer, sourceOffset, destination.buffer, destinationOffset,
            4 * length)
    }

    public copyTexture(source: Texture, destination: Texture, size?: number[])
    {
        size ??= source.size
        this.encoder.copyTextureToTexture({ texture: source.texture }, { texture: destination.texture }, size)
    }

}

interface Pipeline<T extends GPUPipelineBase>
{

    device: Device
    pipeline: T

}

export interface ResourceBindingParams
{

    i: number
    resource: Resource

}

abstract class PassDescriptor<T extends Pipeline<GPUPipelineBase>>
{

    public readonly groups: (GPUBindGroup | null)[]

    protected constructor(public readonly pipeline: T, bindings: ResourceBindingParams[][])
    {
        // Convert resource array to BindGroups
        let { device: { device } } = pipeline
        this.groups = bindings.map((group, i) => group.length === 0 ? null : device.createBindGroup(
        {
            layout: pipeline.pipeline.getBindGroupLayout(i),
            entries: group.map(({ i, resource }) => ({ binding: i, resource: resource.getBinding() }))
        }))
    }

    protected bind(encoder: GPUBindingCommandsMixin)
    {
        // Encode instructions to bind specified resources
        for (let i = 0; i < this.groups.length; i++)
        {
            let group = this.groups[i]
            if (group !== null) encoder.setBindGroup(i, group)
        }
    }

}

export interface VertexFormatParams
{

    format: VertexFormat
    step?: StepMode

}

export interface RenderPipelineParams
{

    vertex?: string
    fragment?: string

    primitive?: Primitive
    cull?: CullMode
    depth?: TextureFormat

    samples?: number
    blend?: boolean

}

export class RenderPipeline implements Pipeline<GPURenderPipeline>
{

    private static getBytes(format: VertexFormat): number
    {
        switch (format)
        {
            case VertexFormat.I32:   case VertexFormat.U32:   case VertexFormat.F32:   return 4
            case VertexFormat.I32_2: case VertexFormat.U32_2: case VertexFormat.F32_2: return 8
            case VertexFormat.I32_3: case VertexFormat.U32_3: case VertexFormat.F32_3: return 12
            case VertexFormat.I32_4: case VertexFormat.U32_4: case VertexFormat.F32_4: return 16
        }
    }

    public readonly pipeline: GPURenderPipeline

    public constructor(public readonly device: Device, shader: Shader, format: TextureFormat,
        vertices: VertexFormatParams[],
    {
        vertex = "vs", fragment = "fs",
        primitive = Primitive.TRIANGLE, cull = CullMode.NONE,
        depth, samples, blend = false
    }: RenderPipelineParams = {})
    {
        // Convert format parameters to layout necessary for initialization
        let entries: GPUVertexBufferLayout[] = vertices.map(({ format, step = StepMode.VERTEX }, i) =>
        ({
            arrayStride: RenderPipeline.getBytes(format),
            stepMode: step,
            attributes:
            [{
                format, offset: 0,
                shaderLocation: i 
            }]
        }))

        this.pipeline = device.device.createRenderPipeline(
        {
            layout: "auto",
            vertex:
            {
                module: shader.module,
                entryPoint: vertex,
                buffers: entries
            },
            fragment:
            {
                module: shader.module,
                entryPoint: fragment,
                targets:
                [{
                    format,
                    blend: blend ? // Configure alpha blend if specified
                    {
                        color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha" },
                        alpha: {}
                    } : undefined
                }]
            },
            primitive:
            {
                topology: primitive,
                cullMode: cull
            },
            depthStencil: depth ? // Configure depth and stencil buffer if specified
            {
                format: depth,
                depthWriteEnabled: true,
                depthCompare: "less"
            } : undefined,
            multisample: { count: samples }
        })
    }

}

export class RenderPass extends PassDescriptor<RenderPipeline>
{

    public readonly index: Buffer | null

    public constructor(pipeline: RenderPipeline, bindings: ResourceBindingParams[][],
        public readonly vertices: Buffer[], index?: Buffer)
    {
        super(pipeline, bindings)
        this.index = index ?? null
    }

    public render(count: number, instances?: number)
    {
        let encoder = this.pipeline.device.renderEncoder

        encoder.setPipeline(this.pipeline.pipeline)
        this.bind(encoder)

        // Bind vertex buffers
        for (let i = 0; i < this.vertices.length; i++)
        {
            let buffer = this.vertices[i].buffer
            encoder.setVertexBuffer(i, buffer)
        }

        // If pass uses an index buffer, bind index buffer
        if (this.index !== null)
        {
            encoder.setIndexBuffer(this.index.buffer, "uint32")
            encoder.drawIndexed(count, instances)
        }
        else encoder.draw(count, instances)
    }

}

export class ComputePipeline implements Pipeline<GPUComputePipeline>
{

    public pipeline: GPUComputePipeline

    public constructor(public readonly device: Device, shader: Shader, entry: string = "main")
    {
        this.pipeline = device.device.createComputePipeline(
        {
            layout: "auto",
            compute: { module: shader.module, entryPoint: entry }
        })
    }

}

export class ComputePass extends PassDescriptor<ComputePipeline>
{

    public constructor(pipeline: ComputePipeline, bindings: ResourceBindingParams[][]) { super(pipeline, bindings) }

    public dispatch(x: number, y: number = 1, z: number = 1)
    {
        let encoder = this.pipeline.device.encoder.beginComputePass()

        encoder.setPipeline(this.pipeline.pipeline)
        this.bind(encoder)

        // Dispatch threads
        encoder.dispatchWorkgroups(x, y, z)
        encoder.end()
    }

}
