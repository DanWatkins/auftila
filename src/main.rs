#[macro_use]
extern crate vulkano;
#[macro_use]
extern crate vulkano_shader_derive;
extern crate vulkano_win;
extern crate winit;

use std::iter;
use std::sync::Arc;
use std::mem;
use vulkano_win::VkSurfaceBuild;
use vulkano::buffer::BufferUsage;
use vulkano::buffer::CpuAccessibleBuffer;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::command_buffer::DynamicState;
use vulkano::device::Device;
use vulkano::framebuffer::Framebuffer;
use vulkano::framebuffer::Subpass;
use vulkano::instance::Instance;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::pipeline::viewport::Viewport;
use vulkano::swapchain;
use vulkano::swapchain::PresentMode;
use vulkano::swapchain::SurfaceTransform;
use vulkano::swapchain::Swapchain;
use vulkano::swapchain::AcquireError;
use vulkano::swapchain::SwapchainCreationError;
use vulkano::sync::now;
use vulkano::sync::GpuFuture;

fn main() {
    let instance = {
        let extensions = vulkano_win::required_extensions();
        Instance::new(None, &extensions, None)
            .expect("failed to create Vulkan instance")
    };

    let physical_device = vulkano::instance::PhysicalDevice::enumerate(&instance).next()
        .expect("no Vulkan device available");

    println!("Using device: {} (type: {:?})", physical_device.name(), physical_device.ty());

    let mut events_loop = winit::EventsLoop::new();
    let window = winit::WindowBuilder::new().build_vk_surface(&events_loop, instance.clone()).unwrap();

    let mut dimensions = {
        let (width, height) = window.window().get_inner_size_pixels().unwrap();
        [width, height]
    };

    let queue = physical_device.queue_families().find(|&q| {
        q.supports_graphics() && window.surface().is_supported(q).unwrap_or(false)
    }).expect("could not find a graphical queue family");

    let (device, mut queues) = {
        let device_ext = vulkano::device::DeviceExtensions {
            khr_swapchain: true,
            .. vulkano::device::DeviceExtensions::none()
        };

        Device::new(physical_device, physical_device.supported_features(), &device_ext, [(queue, 0.5)].iter().cloned())
            .expect("failed to create device")
    };

    let queue = queues.next().unwrap();

    let (mut swapchain, mut images) = {
        let capabilities = window.surface().capabilities(physical_device)
            .expect("failed to get surface capabilities");

        let alpha = capabilities.supported_composite_alpha.iter().next().unwrap();
        let format = capabilities.supported_formats[0].0;

        Swapchain::new(device.clone(), window.surface().clone(), capabilities.min_image_count, format,
                        dimensions, 1, capabilities.supported_usage_flags, &queue,
                        SurfaceTransform::Identity, alpha, PresentMode::Fifo, true, None)
            .expect("failed to create swapchain")
    };

    let vertex_buffer = {
        #[derive(Debug, Clone)]
        struct Vertex { position: [f32; 2] }
        impl_vertex!(Vertex, position);

        CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), [
            Vertex { position: [-0.5, -0.25] },
            Vertex { position: [0.0, 0.5] },
            Vertex { position: [0.25, -0.1] }
        ].iter().cloned())
        .expect("failed to create vertex buffer")
    };

    mod vs {
        #[derive(VulkanoShader)]
        #[ty = "vertex"]
        #[src = "
        #version 450

        layout(location = 0) in vec2 position;

        void main() {
            gl_Position = vec4(position, 0.0, 1.0);
        }
        "]
        struct Dummy;
    }
    
    mod fs {
        #[derive(VulkanoShader)]
        #[ty = "fragment"]
        #[src = "
        #version 450

        layout(location = 0) out vec4 f_color;

        void main() {
            f_color = vec4(1.0, 0.0, 0.0, 1.0);
        }
        "]
        struct Dummy;
    }

    let vs = vs::Shader::load(device.clone()).expect("failed to create vertex shader module");
    let fs = fs::Shader::load(device.clone()).expect("failed to create fragment shader module");

    let render_pass = Arc::new(single_pass_renderpass!(device.clone(),
        attachments: {
            color: {
                load: Clear,
                store: Store,
                format: swapchain.format(),
                samples: 1,
            }
        },
        pass: {
            color: [color],
            depth_stencil: {}
        }).unwrap());

    let pipeline = Arc::new(GraphicsPipeline::start()
        .vertex_input_single_buffer()
        .vertex_shader(vs.main_entry_point(), ())
        .triangle_list()
        .viewports_dynamic_scissors_irrelevant(1)
        .fragment_shader(fs.main_entry_point(), ())
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        .build(device.clone())
        .unwrap());
    
    let mut framebuffers: Option<Vec<Arc<vulkano::framebuffer::Framebuffer<_,_>>>> = None;
    let mut recreate_swapchain = false;
    let mut previous_frame_end = Box::new(now(device.clone())) as Box<GpuFuture>;

    loop {
        previous_frame_end.cleanup_finished();

        if recreate_swapchain {
            dimensions = {
                let (new_width, new_height) = window.window().get_inner_size_pixels().unwrap();
                [new_width, new_height]
            };

            let (new_swapchain, new_images) = match swapchain.recreate_with_dimension(dimensions) {
                Ok(r) => r,
                Err(SwapchainCreationError::UnsupportedDimensions) => {
                    continue;
                },
                Err(err) => panic!("{:?}", err)
            };

            mem::replace(&mut swapchain, new_swapchain);
            mem::replace(&mut images, new_images);

            framebuffers = None;
            recreate_swapchain = false;
        }

        if framebuffers.is_none() {
            let new_framebuffers = Some(images.iter().map(|image| {
                Arc::new(Framebuffer::start(render_pass.clone())
                    .add(image.clone()).unwrap()
                    .build().unwrap())
            }).collect::<Vec<_>>());

            mem::replace(&mut framebuffers, new_framebuffers);
        }

        let (image_num, acquire_future) = match swapchain::acquire_next_image(swapchain.clone(), None) {
            Ok(r) => r,
            Err(AcquireError::OutOfDate) => {
                recreate_swapchain = true;
                continue;
            },
            Err(err) => panic!("{:?}", err)
        };

        // submit draw command
        let command_buffer = match AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family()).unwrap()
            .begin_render_pass(framebuffers.as_ref().unwrap()[image_num].clone(), false, vec![[0.0, 0.0, 1.0, 1.0].into()]).unwrap()
            .draw(pipeline.clone(), DynamicState {
                line_width: None,
                viewports: Some(vec![Viewport {
                    origin: [0.0, 0.0],
                    dimensions: [dimensions[0] as f32, dimensions[1] as f32],
                    depth_range: 0.0 .. 1.0,
                }]),
                scissors: None,
            },
            vertex_buffer.clone(), (), ()).unwrap()
            .end_render_pass().unwrap()
            .build() {
                Ok(cb) => cb,
                Err(err) => panic!("{:?}", err)
            };

        let future = previous_frame_end.join(acquire_future)
            .then_execute(queue.clone(), command_buffer).unwrap()
            .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
            .then_signal_fence_and_flush().unwrap();
        
        previous_frame_end = Box::new(future) as Box<_>;

        // check for window close
        let mut is_done = false;
        events_loop.poll_events(|ev| {
            match ev {
                winit::Event::WindowEvent { event: winit::WindowEvent::Closed, .. } => is_done = true,
                _ => ()
            }
        });

        if (is_done) {
            return;
        }
    }
}