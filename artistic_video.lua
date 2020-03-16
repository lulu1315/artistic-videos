require 'torch'
require 'nn'
require 'image'
require 'loadcaffe'
require 'paths'
require 'artistic_video_core'

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

--preProcess options
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

-- Output options
cmd:option('-print_iter', 0)
cmd:option('-save_iter', 20)
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
    
  gmicbin="/shared/foss-18/gmic-2.8.3_pre/build/gmic"
  
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
  
  -- motion ratio
  if params.output_size ~= 0 then
      motionratio=params.output_size/firstContentImg:size(3)
    else
      motionratio=1
    end
    
  if params.output_size ~= 0 then
    print("up scale ratio: " .. motionratio)
    print("Output image size : " .. params.output_size .. " x " .. firstContentImg:size(2)*motionratio)
  end
  
  -- scale style image
  if params.output_size ~= 0 then
      img_scalex = params.output_size*params.style_scale
      img_scaley = img_scalex/style_ratio
    else
      img_scalex = firstContentImg:size(3)*params.style_scale
      img_scaley = img_scalex/style_ratio
    end

  print("Style image scale: " .. params.style_scale)
  style_image = image.scale(style_image, img_scalex,img_scaley, 'bilinear')
  print("Style image size : " .. style_image:size(3) .. " x " .. style_image:size(2))
  local style_image_caffe = preprocess(style_image):float()
  style_image_caffe = PutOnGPU(style_image_caffe,params)
  
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
  local num_images = params.num_images

  -- create working dir
  workingdir=string.format('%s/work_%s',params.output_folder , params.pid)
  -- workinglog=string.format('2> %s/out.log',workingdir)
  paths.mkdir(workingdir)
    
  -- Iterate over all frames in the video sequence
  print("--------------------------")
  loopstart=params.start_number + params.continue_with - 1
  loopend  =params.start_number + params.continue_with + num_images - 2
  print("-- sequence : " .. loopstart .. " -> " .. loopend)
  print("--------------------------")
  -- for frameIdx=params.start_number + params.continue_with - 1, params.start_number + num_images - 1 do
  -- for frameIdx=params.start_number + params.continue_with - 1, params.continue_with + num_images - 1 do
  for frameIdx=loopstart,loopend do
    framesleft=loopend - frameIdx
    print(colors("%{yellow}\n--------------------------> Working on frame :" .. frameIdx .. " (" .. loopstart .. "->" .. loopend .. ") " .. framesleft .. " frames to go .."))
    print("-- working dir : " .. workingdir)
    -- Set seed
    if params.seed >= 0 then
      torch.manualSeed(params.seed)
    end
    
    wcontent= string.format(params.content_pattern, frameIdx)
    print(string.format('Reading content file "%s".', wcontent))
    wcolor=   string.format('%s/preprocess.png',workingdir)

    
    if params.output_size ~= 0 then
        print("--------------------------> resize content [sizex:" .. params.output_size .. "]")
        resizecmd=string.format('%s -i %s -resize2dx %s,5 -to_colormode 3 -c 0,255 -o %s 2> /var/tmp/artistic.log',gmicbin,wcontent,params.output_size,wcolor);
        -- print (resizecmd)
        os.execute (resizecmd)
    else
        resizecmd=string.format('cp %s %s',wcontent,wcolor)
        -- print (resizecmd)
        os.execute (resizecmd)
    end
    
    -- edges : add edges
    if params.doedges then
        print("----------------------> preprocess edges :" )
        local edges=string.format(params.edges_pattern, frameIdx)
        print(string.format('Reading edge file "%s".', edges))
        local wedges=string.format('%s/edges.png',workingdir)
        local gmic1=''
        local gmic2=''
        local gmic3=''
        local gmic4=''
        if params.edgedilate ~= 0 then
            print("--------------------------> dilate edges [dilate:" .. params.edgedilate .. "]")
            gmic1=string.format("-dilate %s" , params.edgedilate)
        end
        if params.edgesmooth then
            print("--------------------------> smooth edges")
            gmic2="-fx_dreamsmooth 10,0,1,1,0,0.8,0,24,0"
        end
        if params.edgesinvert then
            print("--------------------------> invert edges")
            gmic3="-n 0,1 -oneminus -n 0,255"
        end
        if params.output_size ~= 0 then
            print("--------------------------> resize edges [sizex:" .. params.output_size .. "]")
            gmic4="-resize2dx %s,5"
            gmic4=string.format("-resize2dx %s,5" , params.output_size)
        end
        local gmiccmd=string.format("%s -i %s -to_colormode 3 %s %s %s %s -c 0,255 -o %s 2> /var/tmp/artistic.log" ,gmicbin,edges,gmic1,gmic2,gmic3,gmic4,wedges)
        os.execute(gmiccmd)
        local addedgecmd=string.format("%s -i %s -i %s -blend %s,%s -o %s 2> /var/tmp/artistic.log" ,gmicbin,wcolor,wedges,params.edgesmode,params.edgesopacity,wcolor)
        print("--------------------------> add edges [mode:" .. params.edgesmode .."] [opacity:" .. params.edgesopacity .."]" )
        os.execute(addedgecmd)
    end
    
    -- lce
    if params.lce == 1 then
        print("--------------------------> local contrast enhancement [" .. params.lce .. "]")
        lcecmd=string.format('%s -i %s -fx_LCE 80,0.5,1,1,0,0 -o %s 2> /var/tmp/artistic.log',gmicbin,wcolor,wcolor);
        -- print (lcecmd)
        os.execute (lcecmd)
    end
    
    -- equalize 
    if params.equalize == 1 then
        print("--------------------------> equalize [levels:256 min:" .. params.equalizemin .. " max:" .. params.equalizemax .."]")
        equcmd=string.format('%s -i %s -equalize 256,%s,%s -o %s 2> /var/tmp/artistic.log',gmicbin,wcolor,params.equalizemin,params.equalizemax,wcolor);
         -- print (equcmd)
        os.execute (equcmd)
    end
    
    -- adjustcolor 
    if params.brightness ~= 0 or params.contrast ~= 0 or params.gamma ~= 0 or params.saturation ~= 0 then
        print("--------------------------> adjust color [B/C/G/S:" .. params.brightness .. "," .. params.contrast .. "," .. params.gamma .. "," .. params.saturation .."]")
        adjustcmd=string.format('%s -i %s -fx_adjust_colors %s,%s,%s,0,%s -o %s 2> /var/tmp/artistic.log',gmicbin,wcolor,params.brightness,params.contrast,params.gamma,params.saturation,wcolor);
         -- print (equcmd)
        os.execute (adjustcmd)
    end
    
    -- content blur 
    if params.content_blur ~= 0 then
        print("--------------------------> anisotropic filter content [iter:" .. params.content_blur .. "]")
        -- blurcmd=string.format('%s -i %s -fx_sharp_abstract %s,10,0.5,0,0 -o %s 2> /var/tmp/artistic.log',gmicbin,wcolor,params.content_blur,wcolor);
        blurcmd=string.format('%s -i %s fx_smooth_anisotropic 60,0.7,0.3,0.6,1.1,0.8,30,2,0,1,%s,0,0,50,50 -o %s 2> /var/tmp/artistic.log',gmicbin,wcolor,params.content_blur,wcolor);
         -- print (blurcmd)
        os.execute (blurcmd)
    end
    
    -- noise
    if params.noise ~= 0 then
        print("--------------------------> noise [noise:" .. params.noise .. "]")
        noisecmd=string.format('%s -i %s -fx_simulate_grain 0,1,%s,100,0,0,0,0,0,0,0,0 -o %s 2> /var/tmp/artistic.log',gmicbin,wcolor,params.noise,wcolor);
         -- print (blurcmd)
        os.execute (noisecmd)
    end
    
    -- shave and expand
    if params.shavex ~= 0 or params.shavey ~= 0 or params.expandx ~= 0 or params.expandy ~= 0 then
        img_shave_sizex = firstContentImg:size(3) - (params.shavex*2)
        img_shave_sizey = firstContentImg:size(2) - (params.shavey*2)
        print("after shave : " .. img_shave_sizex .. " x " .. img_shave_sizey)
        img_expand_sizex = firstContentImg:size(3) + (params.expandx*2)
        img_expand_sizey = firstContentImg:size(2) + (params.expandy*2)
        print("after expand : " .. img_expand_sizex .. " x " .. img_expand_sizey)
        -- expand wcolor
        print("--------------------------> shave & expand [shavex/y:" .. img_shave_sizex .. "/" .. img_shave_sizey .. "] [expandx/y:" .. img_expand_sizex .. "/" .. img_expand_sizey .. "]")
        local expandcmd=string.format("%s -i %s -rolling_guidance 8,10,0.5 --resize %s,%s,1,3,0,0,.5,.5,.5,.5 -fill[1] 1,1,1 -resize %s,%s,1,3,0,0,.5,.5,.5,.5 -oneminus[1] -inpaint[0] [1],0 -noise[0] 5,4 -b[1] %s -i %s -mul[0] [1] -resize[2] %s,%s,1,3,0,0,.5,.5,.5,.5 -oneminus[1] -mul[2] [1] -add[0] [2] -remove[-1] -remove[-1] -o %s 2> /var/tmp/artistic.log",gmicbin,wcolor,img_shave_sizex,img_shave_sizey,img_expand_sizex,img_expand_sizey,params.shavex/2,wcolor,img_expand_sizex,img_expand_sizey,wcolor)
        print (expandcmd)
        os.execute(expandcmd)
    end
    
    -- colortransfer : wcolor dans workdir
    if params.docolortransfer == 1 then
        print("--------------------------> color transfert [gmic]")
        -- local colorcmd=string.format('python3 /shared/foss-18/Neural-Tools/linear-color-transfer.py --mode pca --target_image  %s --source_image %s --output_image %s 2> /var/tmp/artistic.log',wcolor,params.style_image,wcolor)
        local colorcmd=string.format('%s -i %s -i %s +transfer_pca[0] [1],ycbcr_y transfer_pca[-1] [1],ycbcr_cbcr -o[2] %s 2> /var/tmp/artistic.log',gmicbin,wcolor,params.style_image,wcolor)
        -- print (colorcmd)
        os.execute (colorcmd)
        -- print (params.doindex)
        if params.doindex == 1 then
            print("--------------------------> color transfert [index:" .. params.indexcolor .. " dither:" .. params.dithering .. "]")
            local indexcmd=string.format('%s %s -colormap %s,%s,1 %s -index[1] [0],%s,1 -remove[0] -fx_sharp_abstract %s,10,0.5,0,0 -o %s 2> /var/tmp/artistic.log',gmicbin,params.style_image,params.indexcolor,params.indexmethod,wcolor,params.dithering,params.indexroll,wcolor)
            print (indexcmd)
            os.execute (indexcmd)
        end
    end
    
    -- histogram transfer
    if params.histogramtransfer == 1 then
        print("--------------------------> histogram transfer")
        histocmd=string.format('%s -i %s -i %s -transfer_histogram[0] [1],512 -o[0] %s 2> /var/tmp/artistic.log',gmicbin,wcolor,params.style_image,wcolor);
        -- print (histocmd)
        os.execute (histocmd)
    end
    
    -- anisotropic smoothing
    if params.anisotropic >= 1 then
        print("--------------------------> anisotropic smoothing [iters:" .. params.anisotropic .. "]")
        anisocmd=string.format('%s -i %s fx_smooth_anisotropic 60,0.7,0.3,0.6,1.1,0.8,30,2,0,1,%s,0,0,50,50 -o %s 2> /var/tmp/artistic.log',gmicbin,wcolor,params.anisotropic,wcolor);
        -- print (anisocmd)
        os.execute (anisocmd)
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
    
    --file names
    woptical= string.format('%s/optical.flo',workingdir)
    wgradient=string.format('%s/gradient.flo',workingdir)
    wtangent= string.format('%s/tangent.flo',workingdir)
    wcustom= string.format('%s/custom.flo',workingdir)
    wopticalexr= string.format('%s/optical.exr',workingdir)
    wgradientexr=string.format('%s/gradient.exr',workingdir)
    wtangentexr= string.format('%s/tangent.exr',workingdir)
    wcustomexr= string.format('%s/custom.exr',workingdir)
    wreliable=string.format('%s/reliable.pgm',workingdir)
    wprewarp=string.format('%s/prewarp.png',workingdir)
    wmask=string.format('%s/mask.png',workingdir)
    
    if frameIdx > params.start_number and params.temporal_weight ~= 0 then
        local prevIndex = frameIdx - 1
        -- preprocess opticalflow
        local flowFileName = getFormatedFlowFileName(params.flow_pattern, math.abs(prevIndex), math.abs(frameIdx))
        print(string.format('Reading flow file "%s".', flowFileName))
        copycmd=string.format('cp %s %s',flowFileName,woptical)
         -- print (copycmd)
        os.execute (copycmd)
        -- resize opticalflow
        if params.output_size ~= 0 then
            print("--------------------------> resize flow file")
            -- local resizeflocmd=string.format("flo2exr %s %s > /var/tmp/artistic.log;gmic -i %s -resize2dx %s,5 -mul %s -o %s 2> /var/tmp/artistic.log;exr2flo %s %s > /var/tmp/artistic.log" ,woptical, wopticalexr,wopticalexr,params.output_size,motionratio,wopticalexr,wopticalexr,woptical)
            local resizeflocmd=string.format("%s -i %s -resize2dx %s,5 -mul %s -o %s 2> /var/tmp/artistic.log",gmicbin,woptical,params.output_size,motionratio,woptical)
            -- print (resizeflocmd)
            os.execute(resizeflocmd)
        end
        -- expand opticalflow
        if params.shavex ~= 0 or params.shavey ~= 0 or params.expandx ~= 0 or params.expandy ~= 0 then
            print("--------------------------> shave & expand [shavex/y:" .. img_shave_sizex .. "/" .. img_shave_sizey .. "] [expandx/y:" .. img_expand_sizex .. "/" .. img_expand_sizey .. "]")
            -- local expandcmd=string.format("flo2exr %s %s;gmic -i %s -resize %s,%s,1,3,0,0,.5,.5,.5,.5 -resize %s,%s,1,3,0,1,.5,.5,.5,.5 -o %s 2> /var/tmp/artistic.log;exr2flo %s %s" ,woptical, wopticalexr,wopticalexr,img_shave_sizex,img_shave_sizey,img_expand_sizex,img_expand_sizey,wopticalexr,wopticalexr,woptical)
            local expandcmd=string.format("%s -i %s -resize %s,%s,1,3,0,0,.5,.5,.5,.5 -resize %s,%s,1,3,0,1,.5,.5,.5,.5 -o %s 2> /var/tmp/artistic.log",gmicbin,woptical,img_shave_sizex,img_shave_sizey,img_expand_sizex,img_expand_sizey,woptical)
            -- print (expandcmd)
            os.execute(expandcmd)
        end
        -- mask
        if params.mask then
            -- copy mask to workdir
            copycmd=string.format('cp %s %s',string.format(params.mask_pattern, params.start_number),wmask)
            print(string.format('Reading mask file "%s".',string.format(params.mask_pattern, params.start_number)))
            os.execute (copycmd)
            -- resize mask
            if params.output_size ~= 0 then
                print("--------------------------> resize mask")
                local resizemaskcmd=string.format("%s -i %s -resize2dx %s,5 -o %s 2> /var/tmp/artistic.log" ,gmicbin,wmask,params.output_size,wmask)
                print (resizemaskcmd)
                os.execute(resizemaskcmd)
            end
            -- multiply opticalflow by mask
            multcmd=string.format("%s -i %s %s -split[1] c -remove[2,3] -div[1] 255 -mul[0] [1] -remove[1] -o %s 2> /var/tmp/artistic.log" ,gmicbin,woptical,wmask,woptical)
            print("--------------------------> mask opticalflow")
            os.execute (multcmd)
        end
            
        -- 
        print(string.format('Reading flow file "%s".', woptical))
        local flow = flowFile.load(woptical)
        -- end opticalflow
        local fileName = build_OutFilename(params, params.start_number+math.abs(prevIndex - params.start_number))
        -- print("warping and blending image : " .. fileName)
        print("warping with motion flow")
        imgWarped = warpImage(image.load(fileName, 3), flow)
        if params.dotangent then
            local tangentFileName = getFormatedFlowFileName(params.tangent_pattern, math.abs(prevIndex), math.abs(frameIdx))
            copycmd=string.format('cp %s %s',tangentFileName,wtangent)
            print(string.format('Reading tangent file "%s".',tangentFileName))
             -- print (copycmd)
            os.execute (copycmd)
            -- expand tangent
            if params.shavex ~= 0 or params.shavey ~= 0 or params.expandx ~= 0 or params.expandy ~= 0 then
                print("--------------------------> shave & expand [shavex/y:" .. img_shave_sizex .. "/" .. img_shave_sizey .. "] [expandx/y:" .. img_expand_sizex .. "/" .. img_expand_sizey .. "]")
                -- local expandcmd=string.format("flo2exr %s %s;gmic -i %s -resize %s,%s,1,3,0,0,.5,.5,.5,.5 -resize %s,%s,1,3,0,1,.5,.5,.5,.5 -o %s 2> /var/tmp/artistic.log;exr2flo %s %s" ,wtangent, wtangentexr,wtangentexr,img_shave_sizex,img_shave_sizey,img_expand_sizex,img_expand_sizey,wtangentexr,wtangentexr,wtangent)
                local expandcmd=string.format("%s -i %s -resize %s,%s,1,3,0,0,.5,.5,.5,.5 -resize %s,%s,1,3,0,1,.5,.5,.5,.5 -o %s 2> /var/tmp/artistic.log",gmicbin,wtangent,img_shave_sizex,img_shave_sizey,img_expand_sizex,img_expand_sizey,wtangent)
                -- print (expandcmd)
                os.execute(expandcmd)
            end
            -- 
            if params.output_size ~= 0 then
                print("--------------------------> resize tangent file")
                -- local resizetangentcmd=string.format("flo2exr %s %s > /var/tmp/artistic.log;gmic -i %s -resize2dx %s,5 -mul %s -o %s 2> /var/tmp/artistic.log;exr2flo %s %s > /var/tmp/artistic.log" ,wtangent, wtangentexr,wtangentexr,params.output_size,motionratio,wtangentexr,wtangentexr,wtangent)
                local resizetangentcmd=string.format("%s -i %s -resize2dx %s,5 -mul %s -o %s 2> /var/tmp/artistic.log",gmicbin,wtangent,params.output_size,motionratio,wtangent)
                -- print (resizetangentcmd)
                os.execute(resizetangentcmd)
            end
            print(string.format('Reading tangent file "%s".', wtangent))
            local mytangent = flowFile.load(wtangent)
            print("warping with tangent flow")
            mytangent:mul(params.tangentbooster)
            print("boost tangent : " .. params.tangentbooster)
            imgWarped = warpImage(imgWarped, mytangent)
        end
        if params.dogradient then
            local gradientFileName = getFormatedFlowFileName(params.gradient_pattern, math.abs(prevIndex), math.abs(frameIdx))
            copycmd=string.format('cp %s %s',gradientFileName,wgradient)
            print(string.format('Reading gradient file "%s".',gradientFileName))
             -- print (copycmd)
            os.execute (copycmd)
            -- expand gradient
            if params.shavex ~= 0 or params.shavey ~= 0 or params.expandx ~= 0 or params.expandy ~= 0 then
                print("--------------------------> shave & expand [shavex/y:" .. img_shave_sizex .. "/" .. img_shave_sizey .. "] [expandx/y:" .. img_expand_sizex .. "/" .. img_expand_sizey .. "]")
                -- local expandcmd=string.format("flo2exr %s %s;gmic -i %s -resize %s,%s,1,3,0,0,.5,.5,.5,.5 -resize %s,%s,1,3,0,1,.5,.5,.5,.5 -o %s 2> /var/tmp/artistic.log;exr2flo %s %s" ,wgradient, wgradientexr,wgradientexr,img_shave_sizex,img_shave_sizey,img_expand_sizex,img_expand_sizey,wgradientexr,wgradientexr,wgradient)
                local expandcmd=string.format("%s -i %s -resize %s,%s,1,3,0,0,.5,.5,.5,.5 -resize %s,%s,1,3,0,1,.5,.5,.5,.5 -o %s 2> /var/tmp/artistic.log",gmicbin,wgradient,img_shave_sizex,img_shave_sizey,img_expand_sizex,img_expand_sizey,wgradient)
                -- print (expandcmd)
                os.execute(expandcmd)
            end
            -- 
            if params.output_size ~= 0 then
                print("--------------------------> resize gradient file")
                -- local resizegradientcmd=string.format("flo2exr %s %s > /var/tmp/artistic.log;gmic -i %s -resize2dx %s,5 -mul %s -o %s 2> /var/tmp/artistic.log;exr2flo %s %s > /var/tmp/artistic.log" ,wgradient, wgradientexr,wgradientexr,params.output_size,motionratio,wgradientexr,wgradientexr,wgradient)
                local resizegradientcmd=string.format("%s -i %s -resize2dx %s,5 -mul %s -o %s 2> /var/tmp/artistic.log",gmicbin,wgradient,params.output_size,motionratio,wgradient)
                -- print (resizetangentcmd)
                os.execute(resizegradientcmd)
            end
            print(string.format('Reading gradient file "%s".', wgradient))
            local mygradient = flowFile.load(wgradient)
            print("warping with gradient flow")
            mygradient:mul(params.gradientbooster)
            print("boost gradient : " .. params.gradientbooster)
            imgWarped = warpImage(imgWarped, mygradient)
        end
        
        if params.docustom then
            local customFileName = getFormatedFlowFileName(params.custom_pattern, math.abs(prevIndex), math.abs(frameIdx))
            copycmd=string.format('cp %s %s',customFileName,wcustom)
            print(string.format('Reading custom file "%s".',customFileName))
             -- print (copycmd)
            os.execute (copycmd)
            -- expand custom
            if params.shavex ~= 0 or params.shavey ~= 0 or params.expandx ~= 0 or params.expandy ~= 0 then
                print("--------------------------> shave & expand [shavex/y:" .. img_shave_sizex .. "/" .. img_shave_sizey .. "] [expandx/y:" .. img_expand_sizex .. "/" .. img_expand_sizey .. "]")
                -- local expandcmd=string.format("flo2exr %s %s;gmic -i %s -resize %s,%s,1,3,0,0,.5,.5,.5,.5 -resize %s,%s,1,3,0,1,.5,.5,.5,.5 -o %s 2> /var/tmp/artistic.log;exr2flo %s %s" ,wcustom, wcustomexr,wcustomexr,img_shave_sizex,img_shave_sizey,img_expand_sizex,img_expand_sizey,wcustomexr,wcustomexr,wcustom)
                local expandcmd=string.format("%s -i %s -resize %s,%s,1,3,0,0,.5,.5,.5,.5 -resize %s,%s,1,3,0,1,.5,.5,.5,.5 -o %s 2> /var/tmp/artistic.log",gmicbin,wcustom,img_shave_sizex,img_shave_sizey,img_expand_sizex,img_expand_sizey,wcustom)
                -- print (expandcmd)
                os.execute(expandcmd)
            end
            -- 
            if params.output_size ~= 0 then
                print("--------------------------> resize custom file")
                -- local resizecustomcmd=string.format("flo2exr %s %s > /var/tmp/artistic.log;gmic -i %s -resize2dx %s,5 -mul %s -o %s 2> /var/tmp/artistic.log;exr2flo %s %s > /var/tmp/artistic.log" ,wcustom, wcustomexr,wcustomexr,params.output_size,motionratio,wcustomexr,wcustomexr,wcustom)
                local resizecustomcmd=string.format("%s -i %s -resize2dx %s,5 -mul %s -o %s 2> /var/tmp/artistic.log",gmicbin,wcustom,params.output_size,motionratio,wcustom)
                -- print (resizecustomcmd)
                os.execute(resizecustomcmd)
            end
            print(string.format('Reading custom file "%s".', wcustom))
            local mycustom = flowFile.load(wcustom)
            print("warping with custom flow")
            mycustom:mul(params.custombooster)
            print("boost custom : " .. params.custombooster)
            imgWarped = warpImage(imgWarped, mycustom)
        end
        
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
        print(string.format('Reading floweights file "%s".', weightsFileName))
        copycmd=string.format('cp %s %s',weightsFileName,wreliable)
         -- print (copycmd)
        os.execute (copycmd)
        -- resize reliable
        if params.output_size ~= 0 then
            print("--------------------------> resize reliable file")
            local resizereliablecmd=string.format("%s -i %s -resize2dx %s,5 -c 0,255 -o %s 2> /var/tmp/artistic.log",gmicbin,wreliable,params.output_size,wreliable)
            -- print (resizereliablecmd)
            os.execute(resizereliablecmd)
        end
        -- expand reliable
        if params.shavex ~= 0 or params.shavey ~= 0 or params.expandx ~= 0 or params.expandy ~= 0 then
            print("--------------------------> shave & expand [shavex/y:" .. img_shave_sizex .. "/" .. img_shave_sizey .. "] [expandx/y:" .. img_expand_sizex .. "/" .. img_expand_sizey .. "]")
            --local expandcmd=string.format("gmic -i %s -div 255 -oneminus -resize %s,%s,1,1,0,0,.5,.5,.5,.5 -resize %s,%s,1,1,0,0,.5,.5,.5,.5 -oneminus -mul 255 -o %s 2> /var/tmp/artistic.log",wreliable,img_shave_sizex,img_shave_sizey,img_expand_sizex,img_expand_sizey,wreliable)
            local expandcmd=string.format("%s -i %s -resize %s,%s,1,1,0,0,.5,.5,.5,.5 -resize %s,%s,1,1,0,1,.5,.5,.5,.5 -o %s 2> /var/tmp/artistic.log",gmicbin,wreliable,img_shave_sizex,img_shave_sizey,img_expand_sizex,img_expand_sizey,wreliable)
            -- print (expandcmd)
            os.execute(expandcmd)
        end
        -- 
        print(string.format('Reading flowWeights file "%s".', wreliable))
        flowWeight = image.load(wreliable):float()
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
    print("-------------> initialisation : " .. init)
    local img = nil
    if init == 'random' then
      img = torch.randn(content_image:size()):float():mul(0.001)
    elseif init == 'image' then
      img = content_image_caffe:clone():float()
    elseif init == 'prevWarped' and frameIdx > params.start_number then
      imgWarped:mul(1-params.content_blend)
      blend_content_image = content_image:clone():mul(params.content_blend)
      imgWarped:add(blend_content_image)
      -- save prewarped image
      image.save(wprewarp, imgWarped)
      img = preprocess(imgWarped):float()
    else
      print('ERROR: Invalid initialization method.')
      os.exit()
    end
    img = PutOnGPU(img, params)
    
    -- Run the optimization to stylize the image, save the result to disk
    runOptimization(params,net,content_losses,style_losses,temporal_losses,img,frameIdx,num_iterations,workingdir,style_image)
    
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


