require 'torch'
require 'nn'
require 'image'
require 'loadcaffe'
require 'paths'
require 'artistic_video_core_light'

local colors = require 'ansicolors'

local flowFile = require 'flowFileLoader'

--------------------------------------------------------------------------------

local cmd = torch.CmdLine()

-- Basic options
cmd:option('-style_image', 'example/seated-nude.jpg','Style target image')
cmd:option('-content_pattern', 'example/marple8_%02d.ppm','Content target pattern')
cmd:option('-num_images', 0, 'Number of content images. Set 0 for autodetect.')
cmd:option('-start_number', 1, 'Frame index to start with')
cmd:option('-continue_with', 1, 'Continue with the given frame index.')
cmd:option('-gpu', 0, 'Zero-indexed ID of the GPU to use; for CPU mode set -gpu = -1')
cmd:option('-number_format', '%04d', 'Number format of the output images.')
cmd:option('-pid', 0)
cmd:option('-output_size', 0)
cmd:option('-content_blur', 0)

--preProcess options
cmd:option('-docolortransfer', 0)
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

-- Output options
cmd:option('-print_iter', 100)
cmd:option('-save_iter', 0)
cmd:option('-output_image', 'out.png')
cmd:option('-output_folder', '')

-- Other options
cmd:option('-style_scale', 1.0)
cmd:option('-pooling', 'max', 'max|avg')
cmd:option('-proto_file', '/shared/foss/artistic-videos/models/VGG_ILSVRC_19_layers_deploy.prototxt')
cmd:option('-model_file', '/shared/foss/artistic-videos/models/VGG_ILSVRC_19_layers.caffemodel')
cmd:option('-seed', -1)
cmd:option('-content_layers', 'relu4_2', 'layers for content')
cmd:option('-style_layers', 'relu1_1,relu2_1,relu3_1,relu4_1,relu5_1', 'layers for style')

-- Advanced options (changing them is usually not required)
cmd:option('-combine_flowWeights_method', 'closestFirst',
           'Which long-term weighting scheme to use: normalize or closestFirst. Default and recommended: closestFirst')



local function main(params)
    
  print("--------------------------> start main")
  
  require 'cutorch'
  require 'cunn'
  cutorch.setDevice(params.gpu + 1)
  require 'cudnn'
  cudnn.benchmark = true
  local loadcaffe_backend = 'cudnn'
  
  print("\n--------------------------> load cnn")
  
  local cnn = loadcaffe.load(params.proto_file, params.model_file, loadcaffe_backend):float()
  cnn = PutOnGPU(cnn, params)

  print("--------------------------> get style image")
  
  local firstContentImg = image.load(string.format(params.content_pattern, params.start_number), 3)
  local style_image_caffe = {}
  local style_image = image.load(params.style_image, 3)
  style_ratio = style_image:size(3)/style_image:size(2)
  print("Content image size : " .. firstContentImg:size(3) .. " x " .. firstContentImg:size(2))
   -- Scale the style image so that it's area equals the area of the content image multiplied by the style scale.
  -- consider output_size
  if params.output_size then
      img_scalex = params.output_size*params.style_scale
      img_scaley = img_scalex/style_ratio
    else
      img_scalex = firstContentImg:size(3)*params.style_scale
      img_scaley = img_scalex/style_ratio
    end
  -- 
  -- local img_scale = customscale * (math.sqrt(firstContentImg:size(2) * firstContentImg:size(3) / (style_image:size(3) * style_image:size(2)))
  --      * params.style_scale)
  print("Style image scalexy: " .. img_scalex .. " " .. img_scaley)
  -- style_image = image.scale(style_image, style_image:size(3) * img_scale, style_image:size(2) * img_scale, 'bilinear')
  style_image = image.scale(style_image, img_scalex,img_scaley, 'bilinear')
  print("Style image size : " .. style_image:size(3) .. " x " .. style_image:size(2))
  local style_image_caffe = preprocess(style_image):float()
  style_image_caffe = PutOnGPU(style_image_caffe,params)
  -- style_image
  -- style_image_caffe (preprocess + PutOnGPU)
  
  print("--------------------------> build net")
  -- Set up the network, inserting style losses. Content and temporal loss will be inserted in each iteration.
  local net, style_losses, losses_indices, losses_type = buildNet(cnn, params, style_image_caffe)
  -- We don't need the base CNN anymore, so clean it up to save memory.
  cnn = nil
  for i=1,#net.modules do
    local module = net.modules[i]
    if torch.type(module) == 'nn.SpatialConvolutionMM' then
        -- remote these, not used, but uses gpu memory
        module.gradWeight = nil
        module.gradBias = nil
    end
  end
  collectgarbage()
  
  -- There can be different setting for the first frame and for subsequent frames.
  local num_iterations_split = params.num_iterations:split(",")
  local numIters_first, numIters_subseq = num_iterations_split[1], num_iterations_split[2] or num_iterations_split[1]
  local init_split = params.init:split(",")
  local init_first, init_subseq = init_split[1], init_split[2] or init_split[1]
  -- local firstImg = nil
  local num_images = params.num_images
  -- local flow_relative_indices_split = params.flow_relative_indices:split(",")

  --create working dir
  workingdir=string.format('%s/work_%s',params.output_folder , params.pid)
  -- print("working dir : " .. workingdir)
  paths.mkdir(workingdir)
    
  -- Iterate over all frames in the video sequence
  for frameIdx=params.start_number + params.continue_with - 1, params.start_number + num_images - 1 do
    print(colors("%{yellow}\n--------------------------> Working on frame :" .. frameIdx))
    -- Set seed
    if params.seed >= 0 then
      torch.manualSeed(params.seed)
    end
    
    wcontent=string.format(params.content_pattern, frameIdx)
    wcolor=string.format('%s/preprocess.png',workingdir)
    if params.output_size then
        print("--------------------------> resize content [sizex:" .. params.output_size .. "]")
        resizecmd=string.format('gmic -i %s -resize2dx %s,5 -o %s',wcontent,params.output_size,wcolor);
        print (resizecmd)
        os.execute (resizecmd)
    else
        resizecmd=string.format('cp %s %s',wcontent,wcolor)
        print (resizecmd)
        os.execute (resizecmd)
    end
    
    -- content blur 
    if params.content_blur then
        print("--------------------------> blur content [blur:" .. params.content_blur .. "]")
        blurcmd=string.format('gmic -i %s -fx_sharp_abstract %s,10,0.5,0,0 -o %s',wcolor,params.content_blur,wcolor);
        print (blurcmd)
        os.execute (blurcmd)
    end
    
    -- colortransfer : wcolor dans workdir , sinon content
    if params.docolortransfer == 3 then
        print("--------------------------> color transfert [neuraltools:pca]")
        local colorcmd=string.format('python3 /shared/foss/Neural-Tools/linear-color-transfer.py --mode pca --target_image  %s --source_image %s --output_image %s',wcolor,params.style_image,wcolor)
        print (colorcmd)
        os.execute (colorcmd)
    end
    if params.docolortransfer == 5 then
        local indexcolor=256
        local dither=0
        print("--------------------------> color transfert [index:" .. indexcolor .. " dither:" .. dither .. "]")
        local colorcmd=string.format('gmic -i %s -i %s -colormap[1] %s,%s,1 -index[0] [1],0,1 -remove[1] -o %s',wcolor,params.style_image,indexcolor,dither,wcolor)
        -- local colorcmd=string.format('python /shared/foss/Neural-Tools/linear-color-transfer.py --mode pca --target_image  %s --source_image %s --output_image %s',wcontent,params.style_image,wcolor)
        print (colorcmd)
        os.execute (colorcmd)
    end
    
    -- edges : add edges
    if params.doedges then
        print("--------------------------> add edges [dilate:" .. params.edgedilate .."] [opacity:" .. params.edgesopacity .."]" )
        local edges=string.format(params.edges_pattern, frameIdx)
        local wedges=string.format('%s/edges.png',workingdir)
        local gmic3=''
        local gmic2=''
        local gmic1=string.format("-dilate %s" , params.edgedilate)
        if params.edgesinvert then
            gmic3="-n 0,1 -oneminus -n 0,255"
        end
        if params.edgesmooth then
            gmic2="-fx_dreamsmooth 10,0,1,1,0,0.8,0,24,0"
        end
        local gmiccmd=string.format("gmic -i %s -to_colormode 3 %s %s %s -o %s 2> /var/tmp/artistic.log" , edges , gmic1 , gmic2 , gmic3 , wedges)
        -- print (gmiccmd)
        os.execute(gmiccmd)
        local addedgecmd=string.format("gmic -i %s -i %s -blend %s,%s -o %s 2> /var/tmp/artistic.log" , wcolor , wedges , params.edgesmode , params.edgesopacity , wcolor)
        -- print (addedgecmd)
        os.execute(addedgecmd)
    end
    
    local fileName = string.format(params.content_pattern, frameIdx)
    if not fileExists(fileName) then return nil end
    content_image = image.load(wcolor, 3)
    print(string.format('Reading content file "%s".', wcolor))
    local content_image_caffe = preprocess(content_image):float()
    content_image_caffe = PutOnGPU(content_image_caffe, params)
    
    local content_losses, temporal_losses = {}, {}
    local additional_layers = 0
    local num_iterations = frameIdx == params.start_number and tonumber(numIters_first) or tonumber(numIters_subseq)
    local init = frameIdx == params.start_number and init_first or init_subseq
    
    if frameIdx > params.start_number and params.temporal_weight ~= 0 then
        local prevIndex = frameIdx - 1
        local flowFileName = getFormatedFlowFileName(params.flow_pattern, math.abs(prevIndex), math.abs(frameIdx))
        print(string.format('Reading flow file "%s".', flowFileName))
        local flow = flowFile.load(flowFileName)
        local fileName = build_OutFilename(params, params.start_number+math.abs(prevIndex - params.start_number))
        -- print("warping and blending image : " .. fileName)
        print("warping with motion flow")
        imgWarped = warpImage(image.load(fileName, 3), flow)
        if params.dotangent then
            local tangentFileName = getFormatedFlowFileName(params.tangent_pattern, math.abs(prevIndex), math.abs(frameIdx))
            print(string.format('Reading tangent file "%s".', tangentFileName))
            local mytangent = flowFile.load(tangentFileName)
            print("warping with tangent flow")
            mytangent:mul(params.tangentbooster)
            print("boost tangent : " .. params.tangentbooster)
            imgWarped = warpImage(imgWarped, mytangent)
        end
        if params.dogradient then
            local gradientFileName = getFormatedFlowFileName(params.gradient_pattern, math.abs(prevIndex), math.abs(frameIdx))
            print(string.format('Reading gradient file "%s".', gradientFileName))
            local mygradient = flowFile.load(gradientFileName)
            print("warping with gradient flow")
            mygradient:mul(params.gradientbooster)
            print("boost gradient : " .. params.gradientbooster)
            imgWarped = warpImage(imgWarped, mygradient)
        end
        -- imgWarped:mul(1-params.content_blend)
        -- blend_content_image = content_image:clone():mul(params.content_blend)
        -- imgWarped:add(blend_content_image)
        imgWarped_caffe = preprocess(imgWarped):float()
        imgWarped_caffe = PutOnGPU(imgWarped_caffe, params)
    end
    
    -- Add content and temporal loss for this iteration. Style loss is already included in the net.
    for i=1, #losses_indices do
      if losses_type[i] == 'content'  then
        print("-------------> content loss")
        local loss_module = getContentLossModuleForLayer(net,
          losses_indices[i] + additional_layers, content_image_caffe, params)
        net:insert(loss_module, losses_indices[i] + additional_layers)
        table.insert(content_losses, loss_module)
        additional_layers = additional_layers + 1
      elseif losses_type[i] == 'prevPlusFlow' and frameIdx > params.start_number then
        print("-------------> prevPlusFlow loss")
        local loss_module = getWeightedContentLossModuleForLayer(net,
            losses_indices[i] + additional_layers, imgWarped_caffe,
            params, nil)
          net:insert(loss_module, losses_indices[i] + additional_layers)
          table.insert(temporal_losses, loss_module)
          additional_layers = additional_layers + 1
      elseif losses_type[i] == 'prevPlusFlowWeighted' and frameIdx > params.start_number then
        -- local flowWeight = {}
        -- Read flow weights
        local weightsFileName = getFormatedFlowFileName(params.flowWeight_pattern , math.abs(frameIdx - 1) , math.abs(frameIdx))
        print(string.format('Reading flowWeights file "%s".', weightsFileName))
        flowWeight = image.load(weightsFileName):float()
        -- Create loss modules
        flowWeight = flowWeight:expand(3, flowWeight:size(2), flowWeight:size(3))
        flowWeight_caffe = PutOnGPU(flowWeight, params)
        print("-------------> prevPlusFlowWeighted loss")
        local loss_module = getWeightedContentLossModuleForLayer(net,
            losses_indices[i] + additional_layers, imgWarped_caffe,
            params, flowWeight_caffe)
        net:insert(loss_module, losses_indices[i] + additional_layers)
        table.insert(temporal_losses, loss_module)
        additional_layers = additional_layers + 1
      end
    end
    
    -- Initialization
    print("--------------------------> initialisation : " .. init)
    local img = nil
    if init == 'random' then
      img = torch.randn(content_image:size()):float():mul(0.001)
    elseif init == 'image' then
      img = content_image_caffe:clone():float()
    elseif init == 'prevWarped' and frameIdx > params.start_number then
      imgWarped:mul(1-params.content_blend)
      blend_content_image = content_image:clone():mul(params.content_blend)
      imgWarped:add(blend_content_image)
      -- imgWarped:cmul(flowWeight:double())
      -- neg_flowWeight=(1-flowWeight):double()
      -- content_image:cmul(neg_flowWeight)
      -- imgWarped:add(content_image)
      -- print("--------------------------> warping and blending : " .. params.content_blend)
      ------
      -- wnewcontent=string.format('%s/newcontent.png',workingdir)
      -- image.save(wnewcontent, imgWarped)
      -- print(string.format('saving newcontent "%s".', wwarp))
      -------
      img = preprocess(imgWarped):float()
    else
      print('ERROR: Invalid initialization method.')
      os.exit()
    end
    img = PutOnGPU(img, params)
    
    -- Run the optimization to stylize the image, save the result to disk
    print("--------------------------> run optimization")
    runOptimization(params, net, content_losses, style_losses, temporal_losses, img, frameIdx, num_iterations)
    
    -- if frameIdx == params.start_number then
    --  print("--------------------------> firstImg")
    --  firstImg = img:clone():float()
    -- end
    
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
  end -- end for
end

local params = cmd:parse(arg)

main(params)


