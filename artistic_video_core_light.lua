require 'optim'

-- modified to include a threshold for relative changes in the loss function as stopping criterion
local lbfgs_mod = require 'lbfgs'

function preprocess(img)
  local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
  local perm = torch.LongTensor{3, 2, 1}
  img = img:index(1, perm):mul(256.0)
  mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
  img:add(-1, mean_pixel)
  return img
end

function deprocess(img)
  local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
  mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
  img = img + mean_pixel
  local perm = torch.LongTensor{3, 2, 1}
  img = img:index(1, perm):div(256.0)
  return img
end

function save_image(img, fileName)
  local disp = deprocess(img:double())
  disp = image.minmax{tensor=disp, min=0, max=1}
  image.save(fileName, disp)
end

function runOptimization(params, net, content_losses, style_losses, temporal_losses,
    img, frameIdx, max_iter,workingdir,style_image)

  -- Run it through the network once to get the proper size for the gradient
  -- All the gradients will come from the extra loss modules, so we just pass
  -- zeros into the top of the net on the backward pass.
  local y = net:forward(img)
  local dy = img.new(#y):zero()

  -- Declaring this here lets us access it in maybe_print
  local optim_state = nil
  if params.optimizer == 'lbfgs' then
    optim_state = {
      maxIter = max_iter,
      tolFunRelative = params.tol_loss_relative,
      tolFunRelativeInterval = params.tol_loss_relative_interval,
      verbose=true,
    }
  elseif params.optimizer == 'adam' then
    optim_state = {
      learningRate = params.learning_rate,
      beta1 = params.beta1,
      epsilon = params.epsilon,
    }
  else
    error(string.format('Unrecognized optimizer "%s"', params.optimizer))
  end

  local function maybe_print(t, loss, alwaysPrint)
    local should_print = (params.print_iter > 0 and t % params.print_iter == 0) or alwaysPrint
    if should_print then
      print(string.format('Iteration %d / %d', t, max_iter))
      for i, loss_module in ipairs(content_losses) do
        print(string.format('  Content %d loss: %f', i, loss_module.loss))
      end
      for i, loss_module in ipairs(temporal_losses) do
        print(string.format('  Temporal %d loss: %f', i, loss_module.loss))
      end
      for i, loss_module in ipairs(style_losses) do
        print(string.format('  Style %d loss: %f', i, loss_module.loss))
      end
      print(string.format('  Total loss: %f', loss))
    end
  end

  local function print_end(t)
    --- calculate total loss
    local loss = 0
    for _, mod in ipairs(content_losses) do
      loss = loss + mod.loss
    end
    for _, mod in ipairs(temporal_losses) do
      loss = loss + mod.loss
    end
    for _, mod in ipairs(style_losses) do
      loss = loss + mod.loss
    end
    -- print informations
    maybe_print(t, loss, true)
  end

  local function maybe_save(t, isEnd)
    local should_save_intermed = params.save_iter > 0 and t % params.save_iter == 0
    local should_save_end = t == max_iter or isEnd
    if should_save_intermed or should_save_end then
      local filename = nil
      if isMultiPass then
        filename = build_OutFilename(params, frameIdx, runIdx)
      else
        filename = build_OutFilename(params, params.start_number+math.abs(frameIdx - params.start_number), should_save_end and -1 or t)
      end
      save_image(img, filename)
    end
  end

  -- Function to evaluate loss and gradient. We run the net forward and
  -- backward to get the gradient, and sum up losses from the loss modules.
  -- optim.lbfgs internally handles iteration and calls this fucntion many
  -- times, so we manually count the number of iterations to handle printing
  -- and saving intermediate results.
  local num_calls = 0
  local function feval(x)
    num_calls = num_calls + 1
    net:forward(x)
    local grad = net:backward(x, dy)
    local loss = 0
    for _, mod in ipairs(content_losses) do
      loss = loss + mod.loss
    end
    for _, mod in ipairs(temporal_losses) do
      loss = loss + mod.loss
    end
    for _, mod in ipairs(style_losses) do
      loss = loss + mod.loss
    end
    maybe_print(num_calls, loss, false)
    -- Only need to print if single-pass algorithm is used.
    if not isMultiPass then 
      maybe_save(num_calls, false)
    end

    collectgarbage()
    -- optim.lbfgs expects a vector for gradients
    return loss, grad:view(grad:nElement())
  end

  start_time = os.time()
  
  -- Run optimization.
  if params.optimizer == 'lbfgs' then
    print('Running optimization with L-BFGS')
    local x, losses = lbfgs_mod.optimize(feval, img, optim_state)
  elseif params.optimizer == 'adam' then
    print('Running optimization with ADAM')
    -- imgmean = torch.mean(img)
    -- print('initial mean value : ' ,imgmean)
    
    for t = 1, max_iter do
      local x, losses = optim.adam(feval, img, optim_state)
      if t % params.save_iter == 0 then
        local interimage    = workingdir .. "/interimage." .. t .. ".png"
        local transferimage = workingdir .. "/transferimage." .. t .. ".png"
        save_image(img, interimage)
        local colorcmd=string.format('python3 /shared/foss-18/Neural-Tools/linear-color-transfer.py --mode pca --target_image  %s --source_image %s --output_image %s 2> tmp.log',interimage,params.style_image,transferimage)
        -- print (colorcmd)
        os.execute (colorcmd)
        newimg = image.load(transferimage, 3)
        img = PutOnGPU(preprocess(newimg):float(), params)
        -- prevmean = imgmean
        -- imgmean = torch.mean(img)
        -- deltamean=torch.abs(imgmean-prevmean)
        -- print('prev mean  : ' ,prevmean)
        -- print('cur mean   : ' ,imgmean)
        -- print('delta mean : ' ,deltamean)
        -- if deltamean < 0.0001 then break end
        if t == params.save_iter then
            freeMemory,totalMemory=cutorch.getMemoryUsage(cutorch.getDevice())
            print('memory(free/total) : ' ,freeMemory*9.3132257461548e-10,totalMemory*9.3132257461548e-10)
        end
        print('iteration : ',t)
      end
    end
  end
  
  end_time = os.time()
  elapsed_time = os.difftime(end_time-start_time)
  print("Running time: " .. elapsed_time .. "s")
  
  print_end(num_calls)
  maybe_save(num_calls, true)
end
function fileExists(name)
   local f=io.open(name,"r")
   if f~=nil then io.close(f) return true else return false end
end

-- warp a given image according to the given optical flow.
-- Disocclusions at the borders will be filled with the VGG mean pixel.
function warpImage(img, flow)
  local mean_pixel = torch.DoubleTensor({123.68/256.0, 116.779/256.0, 103.939/256.0})
  result = image.warp(img, flow, 'bilinear', true, 'pad', -1)
  for x=1, result:size(2) do
    for y=1, result:size(3) do
      if result[1][x][y] == -1 and result[2][x][y] == -1 and result[3][x][y] == -1 then
        result[1][x][y] = mean_pixel[1]
        result[2][x][y] = mean_pixel[2]
        result[3][x][y] = mean_pixel[3]
      end
    end
  end
  return result
end

function getFormatedFlowFileName(pattern, fromIndex, toIndex)
  local flowFileName = pattern
  flowFileName = string.gsub(flowFileName, '{(.-)}',
    function(a) return string.format(a, fromIndex) end )
  flowFileName = string.gsub(flowFileName, '%[(.-)%]',
    function(a) return string.format(a, toIndex) end )
  return flowFileName
end

function build_OutFilename(params, image_number)
  local ext = paths.extname(params.output_image)
  local basename = paths.basename(params.output_image, ext)
  local fileNameBase = '%s%s.' .. params.number_format
  return string.format(fileNameBase .. '.%s',
      params.output_folder, basename, image_number, ext)
end

function PutOnGPU(obj, params)
    return obj:cuda()
end

--
-- LOSS MODULES
--

-- Define an nn Module to compute content loss in-place
local ContentLoss, parent = torch.class('nn.ContentLoss', 'nn.Module')

function ContentLoss:__init(strength, target, normalize)
  parent.__init(self)
  self.strength = strength
  self.target = target
  self.normalize = normalize or false
  self.loss = 0
  self.crit = nn.MSECriterion()
end

function ContentLoss:updateOutput(input)
  if input:nElement() == self.target:nElement() then
    self.loss = self.crit:forward(input, self.target) * self.strength
  else
    print('WARNING: Skipping content loss')
  end
  self.output = input
  return self.output
end

function ContentLoss:updateGradInput(input, gradOutput)
  if input:nElement() == self.target:nElement() then
    self.gradInput = self.crit:backward(input, self.target)
  end
  if self.normalize then
    self.gradInput:div(torch.norm(self.gradInput, 1) + 1e-8)
  end
  self.gradInput:mul(self.strength)
  self.gradInput:add(gradOutput)
  return self.gradInput
end

-- Define an nn Module to compute content loss in-place
local WeightedContentLoss, parent = torch.class('nn.WeightedContentLoss', 'nn.Module')

function WeightedContentLoss:__init(strength, target, weights, normalize, loss_criterion)
  parent.__init(self)
  self.strength = strength
  if weights ~= nil then
    -- Take square root of the weights, because of the way the weights are applied
    -- to the mean square error function. We want w*(error^2), but we can only
    -- do (w*error)^2 = w^2 * error^2
    self.weights = torch.sqrt(weights)
    self.target = torch.cmul(target, self.weights)
  else
    self.target = target
    self.weights = nil
  end
  self.normalize = normalize or false
  self.loss = 0
  if loss_criterion == 'mse' then
    self.crit = nn.MSECriterion()
  elseif loss_criterion == 'smoothl1' then
    self.crit = nn.SmoothL1Criterion()
  else
    print('WARNING: Unknown flow loss criterion. Using MSE.')
    self.crit = nn.MSECriterion()
  end
end

function WeightedContentLoss:updateOutput(input)
  if input:nElement() == self.target:nElement() then
    self.loss = self.crit:forward(input, self.target) * self.strength
    if self.weights ~= nil then
      self.loss = self.crit:forward(torch.cmul(input, self.weights), self.target) * self.strength
    else
      self.loss = self.crit:forward(input, self.target) * self.strength
    end
  else
    print('WARNING: Skipping content loss')
  end
  self.output = input
  return self.output
end

function WeightedContentLoss:updateGradInput(input, gradOutput)
  if input:nElement() == self.target:nElement() then
    if self.weights ~= nil then
      self.gradInput = self.crit:backward(torch.cmul(input, self.weights), self.target)
    else
      self.gradInput = self.crit:backward(input, self.target)
    end
  end
  if self.normalize then
    self.gradInput:div(torch.norm(self.gradInput, 1) + 1e-8)
  end
  self.gradInput:mul(self.strength)
  self.gradInput:add(gradOutput)
  return self.gradInput
end

-- Returns a network that computes the CxC Gram matrix from inputs
-- of size C x H x W
function GramMatrix()
  local net = nn.Sequential()
  net:add(nn.View(-1):setNumInputDims(2))
  local concat = nn.ConcatTable()
  concat:add(nn.Identity())
  concat:add(nn.Identity())
  net:add(concat)
  net:add(nn.MM(false, true))
  return net
end

-- Define an nn Module to compute style loss in-place
local StyleLoss, parent = torch.class('nn.StyleLoss', 'nn.Module')

function StyleLoss:__init(strength, target, normalize)
  parent.__init(self)
  self.normalize = normalize or false
  self.strength = strength
  self.target = target
  self.loss = 0
  
  self.gram = GramMatrix()
  self.G = nil
  self.crit = nn.MSECriterion()
end

function StyleLoss:updateOutput(input)
  self.G = self.gram:forward(input)
  self.G:div(input:nElement())
  self.loss = self.crit:forward(self.G, self.target)
  self.loss = self.loss * self.strength
  self.output = input
  return self.output
end

function StyleLoss:updateGradInput(input, gradOutput)
  local dG = self.crit:backward(self.G, self.target)
  dG:div(input:nElement())
  self.gradInput = self.gram:backward(input, dG)
  if self.normalize then
    self.gradInput:div(torch.norm(self.gradInput, 1) + 1e-8)
  end
  self.gradInput:mul(self.strength)
  self.gradInput:add(gradOutput)
  return self.gradInput
end

local TVLoss, parent = torch.class('nn.TVLoss', 'nn.Module')

function TVLoss:__init(strength)
  parent.__init(self)
  self.strength = strength
  self.x_diff = torch.Tensor()
  self.y_diff = torch.Tensor()
end

function TVLoss:updateOutput(input)
  self.output = input
  return self.output
end

-- TV loss backward pass inspired by kaishengtai/neuralart
function TVLoss:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(input):zero()
  local C, H, W = input:size(1), input:size(2), input:size(3)
  self.x_diff:resize(3, H - 1, W - 1)
  self.y_diff:resize(3, H - 1, W - 1)
  self.x_diff:copy(input[{{}, {1, -2}, {1, -2}}])
  self.x_diff:add(-1, input[{{}, {1, -2}, {2, -1}}])
  self.y_diff:copy(input[{{}, {1, -2}, {1, -2}}])
  self.y_diff:add(-1, input[{{}, {2, -1}, {1, -2}}])
  self.gradInput[{{}, {1, -2}, {1, -2}}]:add(self.x_diff):add(self.y_diff)
  self.gradInput[{{}, {1, -2}, {2, -1}}]:add(-1, self.x_diff)
  self.gradInput[{{}, {2, -1}, {1, -2}}]:add(-1, self.y_diff)
  self.gradInput:mul(self.strength)
  self.gradInput:add(gradOutput)
  return self.gradInput
end

function getContentLossModuleForLayer(net, layer_idx, target_img, params)
  local tmpNet = nn.Sequential()
  for i = 1, layer_idx-1 do
    local layer = net:get(i)
    tmpNet:add(layer)
  end
  local target = tmpNet:forward(target_img):clone()
  local loss_module = nn.ContentLoss(params.content_weight, target, params.normalize_gradients):float()
  loss_module = PutOnGPU(loss_module, params)
  return loss_module
end

function getWeightedContentLossModuleForLayer(net, layer_idx, target_img, params, weights)
  local tmpNet = nn.Sequential()
  for i = 1, layer_idx-1 do
    local layer = net:get(i)
    tmpNet:add(layer)
  end
  local target = tmpNet:forward(target_img):clone()
  local loss_module = nn.WeightedContentLoss(params.temporal_weight, target, weights,
      params.normalize_gradients, params.temporal_loss_criterion):float()
  loss_module = PutOnGPU(loss_module, params)
  return loss_module
end

function buildNet(cnn, params, style_image_caffe)
  
  local content_layers = params.content_layers:split(",")
  local style_layers = params.style_layers:split(",")
  -- Which layer to use for the temporal loss. By default, it uses a pixel based loss, masked by the certainty
  --(indicated by initWeighted).
  local temporal_layers = params.temporal_weight > 0 and {'initWeighted'} or {}
  
  local style_losses = {}
  local contentLike_layers_indices = {}
  local contentLike_layers_type = {}
  
  local next_content_i, next_style_i, next_temporal_i = 1, 1, 1
  local current_layer_index = 1
  local net = nn.Sequential()
  
  -- Set up pixel based loss.
  if temporal_layers[next_temporal_i] == 'init' or temporal_layers[next_temporal_i] == 'initWeighted'  then
    print("Setting up temporal consistency.")
    table.insert(contentLike_layers_indices, current_layer_index)
    table.insert(contentLike_layers_type,
      (temporal_layers[next_temporal_i] == 'initWeighted') and 'prevPlusFlowWeighted' or 'prevPlusFlow')
    next_temporal_i = next_temporal_i + 1
  end
  
  -- Set up other loss modules.
  -- For content loss, only remember the indices at which they are inserted, because the content changes for each frame.
  if params.tv_weight > 0 then
    local tv_mod = nn.TVLoss(params.tv_weight):float()
    tv_mod = PutOnGPU(tv_mod, params) 
    net:add(tv_mod)
    current_layer_index = current_layer_index + 1
  end
  for i = 1, #cnn do
    if next_content_i <= #content_layers or next_style_i <= #style_layers or next_temporal_i <= #temporal_layers then
      local layer = cnn:get(i)
      local name = layer.name
      local layer_type = torch.type(layer)
      local is_pooling = (layer_type == 'cudnn.SpatialMaxPooling' or layer_type == 'nn.SpatialMaxPooling')
      if is_pooling and params.pooling == 'avg' then
        assert(layer.padW == 0 and layer.padH == 0)
        local kW, kH = layer.kW, layer.kH
        local dW, dH = layer.dW, layer.dH
        local avg_pool_layer = nn.SpatialAveragePooling(kW, kH, dW, dH):float()
        avg_pool_layer = PutOnGPU(avg_pool_layer, params)
        local msg = 'Replacing max pooling at layer %d with average pooling'
        print(string.format(msg, i))
        net:add(avg_pool_layer)
      else
        net:add(layer)
      end
      current_layer_index = current_layer_index + 1
      if name == content_layers[next_content_i] then
        print("Setting up content layer", i, ":", layer.name)
        table.insert(contentLike_layers_indices, current_layer_index)
        table.insert(contentLike_layers_type, 'content')
        next_content_i = next_content_i + 1
      end
      if name == temporal_layers[next_temporal_i] then
        print("Setting up temporal layer", i, ":", layer.name)
        table.insert(contentLike_layers_indices, current_layer_index)
        table.insert(contentLike_layers_type, 'prevPlusFlow')
        next_temporal_i = next_temporal_i + 1
      end
      if name == style_layers[next_style_i] then
        print("Setting up style layer  ", i, ":", layer.name)
        local gram = GramMatrix():float()
        gram = PutOnGPU(gram, params)
        local target = nil
        local target_features = net:forward(style_image_caffe):clone()
        local target = gram:forward(target_features):clone()
        target:div(target_features:nElement())
        local norm = params.normalize_gradients
        local loss_module = nn.StyleLoss(params.style_weight, target, norm):float()
        loss_module = PutOnGPU(loss_module, params)
        net:add(loss_module)
        current_layer_index = current_layer_index + 1
        table.insert(style_losses, loss_module)
        next_style_i = next_style_i + 1
      end
    end
  end
  return net, style_losses, contentLike_layers_indices, contentLike_layers_type
end

function buildNetMultiStyle(cnn, params, style_images_caffe)
   -- Handle style blending weights for multiple style inputs
  local style_blend_weights = nil
  if params.style_blend_weights == 'nil' then
    -- Style blending not specified, so use equal weighting
    style_blend_weights = {}
    for i = 1, #style_images_caffe do
      table.insert(style_blend_weights, 1.0)
    end
  else
    style_blend_weights = params.style_blend_weights:split(',')
    assert(#style_blend_weights == #style_images_caffe,
      '-style_blend_weights and -style_images must have the same number of elements')
  end
  -- Normalize the style blending weights so they sum to 1
  local style_blend_sum = 0
  for i = 1, #style_blend_weights do
    style_blend_weights[i] = tonumber(style_blend_weights[i])
    style_blend_sum = style_blend_sum + style_blend_weights[i]
  end
  for i = 1, #style_blend_weights do
    style_blend_weights[i] = style_blend_weights[i] / style_blend_sum
  end
  
  local content_layers = params.content_layers:split(",")
  local style_layers = params.style_layers:split(",")
  -- Which layer to use for the temporal loss. By default, it uses a pixel based loss, masked by the certainty
  --(indicated by initWeighted).
  local temporal_layers = params.temporal_weight > 0 and {'initWeighted'} or {}
  
  local style_losses = {}
  local contentLike_layers_indices = {}
  local contentLike_layers_type = {}
  
  local next_content_i, next_style_i, next_temporal_i = 1, 1, 1
  local current_layer_index = 1
  local net = nn.Sequential()
  
  -- Set up pixel based loss.
  if temporal_layers[next_temporal_i] == 'init' or temporal_layers[next_temporal_i] == 'initWeighted'  then
    print("Setting up temporal consistency.")
    table.insert(contentLike_layers_indices, current_layer_index)
    table.insert(contentLike_layers_type,
      (temporal_layers[next_temporal_i] == 'initWeighted') and 'prevPlusFlowWeighted' or 'prevPlusFlow')
    next_temporal_i = next_temporal_i + 1
  end
  
  -- Set up other loss modules.
  -- For content loss, only remember the indices at which they are inserted, because the content changes for each frame.
  if params.tv_weight > 0 then
    local tv_mod = nn.TVLoss(params.tv_weight):float()
    tv_mod = PutOnGPU(tv_mod, params) 
    net:add(tv_mod)
    current_layer_index = current_layer_index + 1
  end
  for i = 1, #cnn do
    if next_content_i <= #content_layers or next_style_i <= #style_layers or next_temporal_i <= #temporal_layers then
      local layer = cnn:get(i)
      local name = layer.name
      local layer_type = torch.type(layer)
      local is_pooling = (layer_type == 'cudnn.SpatialMaxPooling' or layer_type == 'nn.SpatialMaxPooling')
      if is_pooling and params.pooling == 'avg' then
        assert(layer.padW == 0 and layer.padH == 0)
        local kW, kH = layer.kW, layer.kH
        local dW, dH = layer.dW, layer.dH
        local avg_pool_layer = nn.SpatialAveragePooling(kW, kH, dW, dH):float()
        avg_pool_layer = PutOnGPU(avg_pool_layer, params)
        local msg = 'Replacing max pooling at layer %d with average pooling'
        print(string.format(msg, i))
        net:add(avg_pool_layer)
      else
        net:add(layer)
      end
      current_layer_index = current_layer_index + 1
      if name == content_layers[next_content_i] then
        print("Setting up content layer", i, ":", layer.name)
        table.insert(contentLike_layers_indices, current_layer_index)
        table.insert(contentLike_layers_type, 'content')
        next_content_i = next_content_i + 1
      end
      if name == temporal_layers[next_temporal_i] then
        print("Setting up temporal layer", i, ":", layer.name)
        table.insert(contentLike_layers_indices, current_layer_index)
        table.insert(contentLike_layers_type, 'prevPlusFlow')
        next_temporal_i = next_temporal_i + 1
      end
      if name == style_layers[next_style_i] then
        print("Setting up style layer  ", i, ":", layer.name)
        local gram = GramMatrix():float()
        gram = PutOnGPU(gram, params)
        local target = nil
        for i = 1, #style_images_caffe do
          local target_features = net:forward(style_images_caffe[i]):clone()
          local target_i = gram:forward(target_features):clone()
          target_i:div(target_features:nElement())
          target_i:mul(style_blend_weights[i])
          if i == 1 then
            target = target_i
          else
            target:add(target_i)
          end
        end
        local norm = params.normalize_gradients
        local loss_module = nn.StyleLoss(params.style_weight, target, norm):float()
        loss_module = PutOnGPU(loss_module, params)
        net:add(loss_module)
        current_layer_index = current_layer_index + 1
        table.insert(style_losses, loss_module)
        next_style_i = next_style_i + 1
      end
    end
  end
  return net, style_losses, contentLike_layers_indices, contentLike_layers_type
end
