require 'torch'
require 'nn'
require 'image'
require 'loadcaffe'
require 'paths'
require 'artistic_video_core'
require 'external'

local colors = require 'ansicolors'

local flowFile = require 'flowFileLoader'

--------------------------------------------------------------------------------

local cmd = torch.CmdLine()

-- sequence
cmd:option('-num_images', 0, 'Number of content images. Set 0 for autodetect.')
cmd:option('-start_number', 1, 'Frame index to start with')
cmd:option('-continue_with', 1, 'Continue with the given frame index.')

-- inputs
cmd:option('-style_image', 'example/seated-nude.jpg','Style target image')
cmd:option('-style_scale', 1.0)
cmd:option('-content_pattern', 'example/marple8_%02d.ppm','Content target pattern')
cmd:option('-output_size', 0)
cmd:option('-number_format', '%04d', 'Number format of the output images.')
cmd:option('-output_image', 'out.png')
cmd:option('-output_folder', '')

--lowdef
cmd:option('-style_image_lowdef', 'example/seated-nude.jpg','Style target image')
cmd:option('-style_scale_lowdef', 1.0)
cmd:option('-lowdef', 512)

--preProcess options
cmd:option('-content_blur', 0)
cmd:option('-shavex', 0)
cmd:option('-shavey', 0)
cmd:option('-expandx', 0)
cmd:option('-expandy', 0)
cmd:option('-lce', 0)
cmd:option('-anisotropic', 0)
cmd:option('-equalize', 0)
cmd:option('-equalizemin', '20%')
cmd:option('-equalizemax', '80%')
cmd:option('-brightness', 0)
cmd:option('-contrast', 0)
cmd:option('-gamma', 0)
cmd:option('-saturation', 0)
cmd:option('-noise', 0)
cmd:option('-histogramtransfer', 0)
cmd:option('-docolortransfer', 0)
cmd:option('-doindex', 0)
cmd:option('-indexcolor', 256)
cmd:option('-indexmethod', 1)
cmd:option('-dithering', 1)
cmd:option('-indexroll', 0)
cmd:option('-doedges', false)
cmd:option('-edges_pattern' , 'example/edges.%02d.png','edges pattern')
cmd:option('-edgedilate', 0)
cmd:option('-edgesmooth', false)
cmd:option('-edgesopacity', 1)
cmd:option('-edgesmode', 'subtract')
cmd:option('-edgesinvert', false)
cmd:option('-dogradient', false)
cmd:option('-gradient_pattern' , 'example/tangent.%02d.png','tangent pattern')
cmd:option('-gradientbooster', 1)
cmd:option('-dotangent', false)
cmd:option('-tangent_pattern' , 'example/tangent.%02d.png','tangent pattern')
cmd:option('-tangentbooster', 1)
cmd:option('-docustom', false)
cmd:option('-custom_pattern' , 'example/custom.%02d.png','custom pattern')
cmd:option('-custombooster', 1)
cmd:option('-mask', false)
cmd:option('-mask_pattern' , 'example/mask.%02d.png','mask pattern')

--Flow options
cmd:option('-flow_pattern', 'example/deepflow/backward_[%d]_{%d}.flo','Optical flow files pattern')
cmd:option('-flowWeight_pattern', 'example/deepflow/reliable_[%d]_{%d}.pgm','Optical flow weight files pattern.')

-- Optimization options
cmd:option('-content_weight', 5e0)
cmd:option('-content_blend', 5e-1)
cmd:option('-style_weight', 1e2)
cmd:option('-temporal_weight', 1e3)
cmd:option('-tv_weight', 1e-3)
cmd:option('-temporal_loss_criterion', 'mse', 'mse|smoothl1')
cmd:option('-num_iterations', '2000,1000',
           'Can be set separately for the first and for subsequent iterations, separated by comma, or one value for all.')
cmd:option('-tol_loss_relative', 0.0001, 'Stop if relative change of the loss function is below this value')
cmd:option('-tol_loss_relative_interval', 50, 'Interval between two loss comparisons')
cmd:option('-normalize_gradients', false)
cmd:option('-init', 'image,prevWarped', 'random|image,random|image|prev|prevWarped')
cmd:option('-optimizer', 'lbfgs', 'lbfgs|adam')
cmd:option('-learning_rate', 1e1)
cmd:option('-beta1', 9e-1)
cmd:option('-epsilon', 1e-8)

-- Other options
cmd:option('-print_iter', 0)
cmd:option('-save_iter', 20)
cmd:option('-gpu', 0, 'Zero-indexed ID of the GPU to use; for CPU mode set -gpu = -1')
cmd:option('-pid', 0)
cmd:option('-pooling', 'max', 'max|avg')
cmd:option('-proto_file', '/shared/foss/artistic-videos/models/VGG_ILSVRC_19_layers_deploy.prototxt')
cmd:option('-model_file', '/shared/foss/artistic-videos/models/VGG_ILSVRC_19_layers.caffemodel')
cmd:option('-seed', -1)
cmd:option('-content_layers', 'relu4_2', 'layers for content')
cmd:option('-style_layers', 'relu1_1,relu2_1,relu3_1,relu4_1,relu5_1', 'layers for style')

-- Advanced options (changing them is usually not required)
cmd:option('-combine_flowWeights_method', 'closestFirst',
           'Which long-term weighting scheme to use: normalize or closestFirst. Default and recommended: closestFirst')


local round = function(a)
    return math.floor(a + 0.5) -- where prec is 10^n, starting at 0
end

local function main(params)
  -- print("--------------------------> start main")
  require 'cutorch'
  require 'cunn'
  cutorch.setDevice(params.gpu + 1)
  require 'cudnn'
  cudnn.benchmark = true
  local loadcaffe_backend = 'cudnn'
      -- Set seed
    if params.seed >= 0 then
      torch.manualSeed(params.seed)
    end
-- There can be different setting for the first frame and for subsequent frames.
  local num_iterations_split = params.num_iterations:split(",")
  local numIters_first, numIters_subseq = num_iterations_split[1], num_iterations_split[2] or num_iterations_split[1]
  local init_split = params.init:split(",")
  local init_first, init_subseq = init_split[1], init_split[2] or init_split[1]
  local num_images = params.num_images
  
  print("\n\n--------------------------> load cnn")
  local cnn = loadcaffe.load(params.proto_file, params.model_file, loadcaffe_backend):float()
  cnn = PutOnGPU(cnn, params)

  print("\n--------------------------> get first content image")
  local firstContentImg = image.load(string.format(params.content_pattern, params.start_number), 3)
  content_ratio = firstContentImg:size(3)/firstContentImg:size(2)
  print("Content image size : " .. firstContentImg:size(3) .. " x " .. firstContentImg:size(2) .. " ratio : " .. content_ratio)
  
  print("--------------------------> get hi def style image")
  local style_image = image.load(params.style_image, 3)
  style_ratio = style_image:size(3)/style_image:size(2)
  if params.output_size ~= 0 then
      img_scalex = params.output_size*params.style_scale
    else
      img_scalex = firstContentImg:size(3)*params.style_scale
    end
  img_scaley = img_scalex/style_ratio
  style_image = image.scale(style_image,img_scalex,img_scaley,'bilinear')
  print("Style hidef image size : " .. style_image:size(3) .. " x " .. style_image:size(2) .. " scale : " .. params.style_scale)
  
  if params.lowdef ~= 0 then
    print("--------------------------> get low def style image")
    style_image_lowdef = image.load(params.style_image_lowdef, 3)
    style_ratio_lowdef = style_image_lowdef:size(3)/style_image_lowdef:size(2)
    img_scalex_lowdef = params.lowdef*params.style_scale_lowdef
    img_scaley_lowdef = img_scalex_lowdef/style_ratio_lowdef
    style_image_lowdef = image.scale(style_image_lowdef, img_scalex_lowdef,img_scaley_lowdef, 'bilinear')
    print("Style lowdef image size : " .. style_image_lowdef:size(3) .. " x " .. style_image_lowdef:size(2) .. " scale : " .. params.style_scale_lowdef)
  end
  
  print("\n--------------------------> build final net")
  local style_image_caffe = {}
  local style_image_caffe = preprocess(style_image):float()
  style_image_caffe = PutOnGPU(style_image_caffe,params)
  -- Set up the network, inserting style losses. Content and temporal loss will be inserted in each iteration.
  local net, style_losses, losses_indices, losses_type = buildNet(cnn, params, style_image_caffe)
  for i=1,#net.modules do
    local module = net.modules[i]
    if torch.type(module) == 'nn.SpatialConvolutionMM' then
        -- remove these, not used, but uses gpu memory
        module.gradWeight = nil
        module.gradBias = nil
    end
  end
  
if params.lowdef ~= 0 then
  print("\n--------------------------> build lowdef net")
  local style_image_caffe_lowdef = {}
  local style_image_caffe_lowdef = preprocess(style_image_lowdef):float()
  style_image_caffe_lowdef = PutOnGPU(style_image_caffe_lowdef,params)
  -- Set up the network, inserting style losses. Content and temporal loss will be inserted in each iteration.
  net_lowdef, style_losses_lowdef, losses_indices_lowdef, losses_type_lowdef = buildNet(cnn, params, style_image_caffe_lowdef)
    for i=1,#net_lowdef.modules do
    local modulelowdef = net_lowdef.modules[i]
    if torch.type(modulelowdef) == 'nn.SpatialConvolutionMM' then
        -- remove these, not used, but uses gpu memory
        modulelowdef.gradWeight = nil
        modulelowdef.gradBias = nil
    end
  end
end

  -- We don't need the base CNN anymore, so clean it up to save memory.
  cnn = nil
  collectgarbage()
  
-- create working dir
  workingdir=string.format('%s/work_%s',params.output_folder , params.pid)
  paths.mkdir(workingdir)
    
  -- print("\n--------------------------")
  loopstart=params.start_number + params.continue_with - 1
  loopend  =params.start_number + params.continue_with + num_images - 2
  -- print("-- sequence : " .. loopstart .. " -> " .. loopend)
  -- print("--------------------------")

  -- Iterate over all frames in the video sequence
  for frameIdx=loopstart,loopend do
    
    framesleft=loopend - frameIdx
    print(colors("%{yellow}\n--------------------------> Working on frame : " .. frameIdx .. " (" .. loopstart .. "->" .. loopend .. ") " .. framesleft .. " frames to go .."))
    print("working dir : " .. workingdir)

    -- preprocess color content
    content_image=preprocess_color(params,frameIdx,workingdir,firstContentImg:size(3),firstContentImg:size(2))
    -- preprocess flows fields
    if frameIdx > params.start_number and params.temporal_weight ~= 0 then
        imgWarped,flowWeight = preprocess_flow(params,frameIdx,workingdir,firstContentImg:size(3),firstContentImg:size(2),content_ratio)
    end
    
    local num_iterations = frameIdx == params.start_number and tonumber(numIters_first) or tonumber(numIters_subseq)
    local init = frameIdx == params.start_number and init_first or init_subseq
    -- print ("num iterations : " .. num_iterations .. " init : " .. init)
    
-- LOWDEF LOWDEF LOWDEF
if params.lowdef ~= 0 then
    print (colors("%{blue}\n--------------------------> lowdef pass"))
    -- resize images
    lowdef_ratio=params.lowdef/params.output_size
    content_ratio = content_image:size(3)/content_image:size(2)
    xlowdef=round((params.output_size + (params.expandx*2))*lowdef_ratio)
    ylowdef=round(xlowdef/content_ratio)
    print("lowdef size : " .. xlowdef .. "x" .. ylowdef .. " lowdef ratio : " .. lowdef_ratio)
    -- readjust if expand
    content_image_lowdef = image.scale(content_image,xlowdef,ylowdef,'bilinear')
    -- loading/initializing optimisation images
    local content_image_lowdef_caffe = preprocess(content_image_lowdef):float()
    content_image_lowdef_caffe = PutOnGPU(content_image_lowdef_caffe, params)
    if frameIdx > params.start_number and params.temporal_weight ~= 0 then
        imgWarped_lowdef = image.scale(imgWarped,xlowdef,ylowdef,'bilinear')
        imgWarped_lowdef_caffe = preprocess(imgWarped_lowdef):float()
        imgWarped_lowdef_caffe = PutOnGPU(imgWarped_lowdef_caffe, params)
        flowWeight_lowdef = image.scale(flowWeight,xlowdef,ylowdef,'bilinear')
        flowWeight_lowdef = flowWeight_lowdef:expand(3, flowWeight_lowdef:size(2), flowWeight_lowdef:size(3))
        flowWeight_lowdef_caffe = PutOnGPU(flowWeight_lowdef, params)
    end
    
    -- Add content and temporal loss for this iteration. Style loss is already included in the net.
    local content_losses_lowdef, temporal_losses_lowdef = {}, {}
    local additional_layers_lowdef = 0
    
    for i=1, #losses_indices_lowdef do
      if losses_type_lowdef[i] == 'content'  then
        -- print("-------------> content lowdef loss")
        local loss_module_lowdef = getContentLossModuleForLayer(net_lowdef,
          losses_indices_lowdef[i] + additional_layers_lowdef, content_image_lowdef_caffe, params)
        net_lowdef:insert(loss_module_lowdef, losses_indices_lowdef[i] + additional_layers_lowdef)
        table.insert(content_losses_lowdef, loss_module_lowdef)
        additional_layers_lowdef = additional_layers_lowdef + 1
      elseif losses_type_lowdef[i] == 'prevPlusFlow' and frameIdx > params.start_number then
        -- print("-------------> prevPlusFlow lowdef loss")
        local loss_module_lowdef = getWeightedContentLossModuleForLayer(net_lowdef,
            losses_indices_lowdef[i] + additional_layers_lowdef, imgWarped_lowdef_caffe,
            params, nil)
          net_lowdef:insert(loss_module_lowdef, losses_indices_lowdef[i] + additional_layers_lowdef)
          table.insert(temporal_losses_lowdef, loss_module_lowdef)
          additional_layers_lowdef = additional_layers_lowdef + 1
      elseif losses_type_lowdef[i] == 'prevPlusFlowWeighted' and frameIdx > params.start_number then
        -- print("-------------> prevPlusFlowWeighted lowdef loss")
        local loss_module_lowdef = getWeightedContentLossModuleForLayer(net_lowdef,
            losses_indices_lowdef[i] + additional_layers_lowdef, imgWarped_lowdef_caffe,
            params, flowWeight_lowdef_caffe)
        net_lowdef:insert(loss_module_lowdef, losses_indices_lowdef[i] + additional_layers_lowdef)
        table.insert(temporal_losses_lowdef, loss_module_lowdef)
        additional_layers_lowdef = additional_layers_lowdef + 1
      end
    end
    
    -- Initialization
    -- print("-------------> initialisation lowdef : " .. init)
    local img = nil
    if init == 'random' then
      img = torch.randn(content_image_lowdef:size()):float():mul(0.001)
    elseif init == 'image' then
      img = content_image_lowdef_caffe:clone():float()
    elseif init == 'prevWarped' and frameIdx > params.start_number then
      img = preprocess(imgWarped_lowdef):float()
    else
      print('ERROR: Invalid initialization method.')
      os.exit()
    end
    img = PutOnGPU(img, params)
    
        -- Run the optimization to stylize the image, save the result to disk
    runOptimization(params,net_lowdef,content_losses_lowdef,style_losses_lowdef,temporal_losses_lowdef,img,frameIdx,num_iterations,workingdir,style_image_lowdef,0)
    
    -- Remove this iteration's content and temporal layers
    for i=#losses_indices_lowdef, 1, -1 do
      if frameIdx > params.start_number or losses_type_lowdef[i] == 'content' then
        if losses_type_lowdef[i] == 'prevPlusFlowWeighted' or losses_type_lowdef[i] == 'prevPlusFlow' then
          additional_layers_lowdef = additional_layers_lowdef - 1
          net_lowdef:remove(losses_indices_lowdef[i] + additional_layers_lowdef)
        else
          additional_layers_lowdef = additional_layers_lowdef - 1
          net_lowdef:remove(losses_indices_lowdef[i] + additional_layers_lowdef)
        end
      end
    end
    
    -- Ensure that all layer have been removed correctly
    assert(additional_layers_lowdef == 0)
    
-- HIDEF HIDEF HIDEF
    print (colors("%{blue}\n--------------------------> hidef pass"))
    
    -- prepare images for hi def 
    fileName_lowdef = build_OutFilename(params, frameIdx, 0)
    print ("reading lowdef file : " .. fileName_lowdef )
    result_lowdef = image.load(fileName_lowdef, 3)
    print ("upscaling lowdef file : " .. params.output_size + (params.expandx*2) .. "/" .. round((params.output_size+ (params.expandx*2))/content_ratio) )
    result_lowdef_upscale = image.scale(result_lowdef,params.output_size + (params.expandx*2),round((params.output_size+ (params.expandx*2))/content_ratio),'bilinear')
    print ("blending lowdef/content : " .. params.content_blend )
    result_lowdef_upscale:mul(1-params.content_blend)
    content_image:mul(params.content_blend)
    content_image:add(result_lowdef_upscale)
    wpreprocess_withlowdef=string.format('%s/preprocess_withlowdef.png',workingdir)
    print ("writing preprocess with lowdef file : " .. wpreprocess_withlowdef )
    image.save(wpreprocess_withlowdef, content_image)
end

    -- loading/initializing optimisation images
    local content_image_caffe = preprocess(content_image):float()
    content_image_caffe = PutOnGPU(content_image_caffe, params)
    if frameIdx > params.start_number and params.temporal_weight ~= 0 then
        imgWarped_caffe = preprocess(imgWarped):float()
        imgWarped_caffe = PutOnGPU(imgWarped_caffe, params)
        flowWeight = flowWeight:expand(3, flowWeight:size(2), flowWeight:size(3))
        flowWeight_caffe = PutOnGPU(flowWeight, params)
    end
    
    -- Add content and temporal loss for this iteration. Style loss is already included in the net.
    local content_losses, temporal_losses = {}, {}
    local additional_layers = 0
    
    for i=1, #losses_indices do
      if losses_type[i] == 'content'  then
        -- print("-------------> content loss")
        local loss_module = getContentLossModuleForLayer(net,
          losses_indices[i] + additional_layers, content_image_caffe, params)
        net:insert(loss_module, losses_indices[i] + additional_layers)
        table.insert(content_losses, loss_module)
        additional_layers = additional_layers + 1
      elseif losses_type[i] == 'prevPlusFlow' and frameIdx > params.start_number then
        -- print("-------------> prevPlusFlow loss")
        local loss_module = getWeightedContentLossModuleForLayer(net,
            losses_indices[i] + additional_layers, imgWarped_caffe,
            params, nil)
          net:insert(loss_module, losses_indices[i] + additional_layers)
          table.insert(temporal_losses, loss_module)
          additional_layers = additional_layers + 1
      elseif losses_type[i] == 'prevPlusFlowWeighted' and frameIdx > params.start_number then
        -- print("-------------> prevPlusFlowWeighted loss")
        local loss_module = getWeightedContentLossModuleForLayer(net,
            losses_indices[i] + additional_layers, imgWarped_caffe,
            params, flowWeight_caffe)
        net:insert(loss_module, losses_indices[i] + additional_layers)
        table.insert(temporal_losses, loss_module)
        additional_layers = additional_layers + 1
      end
    end
    
    -- Initialization
    -- print("-------------> initialisation : " .. init)
    local img = nil
    if init == 'random' then
      img = torch.randn(content_image:size()):float():mul(0.001)
    elseif init == 'image' then
      img = content_image_caffe:clone():float()
    elseif init == 'prevWarped' and frameIdx > params.start_number then
      img = preprocess(imgWarped):float()
    else
      print('ERROR: Invalid initialization method.')
      os.exit()
    end
    img = PutOnGPU(img, params)
    
    -- Run the optimization to stylize the image, save the result to disk
    runOptimization(params,net,content_losses,style_losses,temporal_losses,img,frameIdx,num_iterations,workingdir,style_image,1)
    
    -- Remove this iteration's content and temporal layers
    for i=#losses_indices, 1, -1 do
      if frameIdx > params.start_number or losses_type[i] == 'content' then
        if losses_type[i] == 'prevPlusFlowWeighted' or losses_type[i] == 'prevPlusFlow' then
          additional_layers = additional_layers - 1
          net:remove(losses_indices[i] + additional_layers)
        else
          additional_layers = additional_layers - 1
          net:remove(losses_indices[i] + additional_layers)
        end
      end
    end
    
    -- Ensure that all layer have been removed correctly
    assert(additional_layers == 0)
    
    --if params.shavex ~= 0 or params.shavey ~= 0 or params.expandx ~= 0 or params.expandy ~= 0 then
    -- crop back final frame
    --    local fileName = build_OutFilename(params, frameIdx, 1)
    --    print(string.format('%s', fileName))
    --    resizefinalcmd=string.format("gmic -i %s -resize %s,%s,1,3,0,0,.5,.5,.5,.5 -o %s",string.format('%s', fileName),firstContentImg:size(3),firstContentImg:size(2),string.format('%s', fileName))
    --    os.execute(resizefinalcmd)
    --end
  end -- end for
  
end

local params = cmd:parse(arg)

main(params)


