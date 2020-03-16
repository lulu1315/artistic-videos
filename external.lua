require 'artistic_video_core'
local colors = require 'ansicolors'

function preprocess_color(params,frameIdx,workingdir,sizex,sizey)
    
    print(colors("%{blue}\n--------------------------> preprocessing content"))
    
    local gmicbin="/shared/foss-18/gmic-2.8.3_pre/build/gmic"
    
    wcontent= string.format(params.content_pattern, frameIdx)
    wcolor=   string.format('%s/preprocess.png',workingdir)
    wgradientmask=string.format('%s/gradientmask.png',workingdir)
    print(string.format('Reading content file "%s".', wcontent))
    
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
        print("--------------------------> preprocess edges :" )
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
        local gmiccmd=string.format("%s -i %s -to_colormode 3 %s %s %s %s -c 0,255 -o %s 2> /var/tmp/artistic.log" ,gmicbin,edges,gmic3,gmic1,gmic2,gmic4,wedges)
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
        if params.output_size == 0 then
            gradcmd=string.format('%s -i %s -b 2 -luminance -gradient_norm -le .5 mul 255 -c 0,255 -o %s 2> /var/tmp/artistic.log',gmicbin,wcontent,wgradientmask);
        else
            gradcmd=string.format('%s -i %s -b 2 -luminance -gradient_norm -le .5 mul 255 -resize2dx %s,5 -c 0,255 -o %s 2> /var/tmp/artistic.log',gmicbin,wcontent,params.output_size,wgradientmask);
        end
        -- print (gradcmd)
        os.execute (gradcmd)
        noisecmd=string.format('%s -i %s %s -n[1] 0,1 --oneminus[1] --fx_simulate_grain[0] 0,1,%s,100,0,0,0,0,0,0,0,0 -mul[0] [2] -mul[3] [1] -add[0] [3] -o[0] %s 2> /var/tmp/artistic.log',gmicbin,wcolor,wgradientmask,params.noise,wcolor);
        -- print (noisecmd)
        os.execute (noisecmd)
    end
    
    -- shave and expand
    if params.shavex ~= 0 or params.shavey ~= 0 or params.expandx ~= 0 or params.expandy ~= 0 then
        img_shave_sizex = sizex - (params.shavex*2)
        img_shave_sizey = sizey - (params.shavey*2)
        img_expand_sizex = sizex + (params.expandx*2)
        img_expand_sizey = sizey + (params.expandy*2)
        -- print("shave : " .. img_shave_sizex .. " x " .. img_shave_sizey .. "expand : " .. img_expand_sizex .. " x " .. img_expand_sizey)
        -- expand wcolor
        print("--------------------------> shave & expand content [shavex/y:" .. img_shave_sizex .. "/" .. img_shave_sizey .. "] [expandx/y:" .. img_expand_sizex .. "/" .. img_expand_sizey .. "]")
        local expandcmd=string.format("%s -i %s -rolling_guidance 8,10,0.5 --resize %s,%s,1,3,0,0,.5,.5,.5,.5 -fill[1] 1,1,1 -resize %s,%s,1,3,0,0,.5,.5,.5,.5 -oneminus[1] -inpaint[0] [1],0 -noise[0] 5,4 -b[1] %s -i %s -mul[0] [1] -resize[2] %s,%s,1,3,0,0,.5,.5,.5,.5 -oneminus[1] -mul[2] [1] -add[0] [2] -remove[-1] -remove[-1] -o %s 2> /var/tmp/artistic.log",gmicbin,wcolor,img_shave_sizex,img_shave_sizey,img_expand_sizex,img_expand_sizey,params.shavex/2,wcolor,img_expand_sizex,img_expand_sizey,wcolor)
        -- print (expandcmd)
        os.execute(expandcmd)
    end
    
    -- colortransfer : wcolor dans workdir
    if params.docolortransfer == 1 then
        print("--------------------------> color transfert [gmic]")
        -- local colorcmd=string.format('python3 /shared/foss-18/Neural-Tools/linear-color-transfer.py --mode pca --target_image  %s --source_image %s --output_image %s 2> /var/tmp/artistic.log',wcolor,params.style_image,wcolor)
        local colorcmd=string.format('%s -i %s -i %s +transfer_pca[0] [1],ycbcr_y transfer_pca[-1] [1],ycbcr_cbcr -o[2] %s 2> /var/tmp/artistic.log',gmicbin,wcolor,params.style_image_lowdef,wcolor)
        -- print (colorcmd)
        os.execute (colorcmd)
        -- print (params.doindex)
        if params.doindex == 1 then
            print("--------------------------> color transfert [index:" .. params.indexcolor .. " dither:" .. params.dithering .. "]")
            local indexcmd=string.format('%s %s -colormap %s,%s,1 %s -index[1] [0],%s,1 -remove[0] -fx_sharp_abstract %s,10,0.5,0,0 -o %s 2> /var/tmp/artistic.log',gmicbin,params.style_image_lowdef,params.indexcolor,params.indexmethod,wcolor,params.dithering,params.indexroll,wcolor)
            print (indexcmd)
            os.execute (indexcmd)
        end
    end
    
    -- histogram transfer
    if params.histogramtransfer == 1 then
        print("--------------------------> histogram transfer")
        histocmd=string.format('%s -i %s -i %s -transfer_histogram[0] [1],512 -o[0] %s 2> /var/tmp/artistic.log',gmicbin,wcolor,params.style_image_lowdef,wcolor);
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
    content_image = image.load(wcolor, 3)
    print(string.format('Reading content file "%s".', wcolor))
    return content_image
end

local round = function(a)
    return math.floor(a + 0.5) -- where prec is 10^n, starting at 0
end

function preprocess_flow(params,frameIdx,workingdir,sizex,sizey,content_ratio)
    local flowFile = require 'flowFileLoader'
    local gmicbin="/shared/foss-18/gmic-2.8.3_pre/build/gmic"
    print(colors("%{blue}\n--------------------------> preprocessing flow files"))
    --file names
    wcolor      =string.format('%s/preprocess.png',workingdir)
    woptical    =string.format('%s/optical.flo',workingdir)
    wgradient   =string.format('%s/gradient.flo',workingdir)
    wtangent    =string.format('%s/tangent.flo',workingdir)
    wcustom     =string.format('%s/custom.flo',workingdir)
    wopticalexr =string.format('%s/optical.exr',workingdir)
    wgradientexr=string.format('%s/gradient.exr',workingdir)
    wtangentexr =string.format('%s/tangent.exr',workingdir)
    wcustomexr  =string.format('%s/custom.exr',workingdir)
    wreliable   =string.format('%s/reliable.pgm',workingdir)
    wmask       =string.format('%s/mask.png',workingdir)
    
        local prevIndex = frameIdx - 1
        local flowFileName = getFormatedFlowFileName(params.flow_pattern, math.abs(prevIndex), math.abs(frameIdx))
        local FirstflowFile = flowFile.load(flowFileName)
        print(string.format('Reading flow file "%s".', flowFileName))
        motionratio=params.output_size/FirstflowFile:size(3)
        print("motion ratio : " .. motionratio)
        -- preprocess opticalflow
        copycmd=string.format('cp %s %s',flowFileName,woptical)
         -- print (copycmd)
        os.execute (copycmd)
        
        -- resize opticalflow
        if params.output_size ~= 0 then
            print("--------------------------> resize opticalflow")
            local resizeflocmd=string.format("%s -i %s -resize2dx %s,5 -mul %s -o %s 2> /var/tmp/artistic.log",gmicbin,woptical,params.output_size,motionratio,woptical)
            -- print (resizeflocmd)
            os.execute(resizeflocmd)
        end
        -- expand opticalflow
        if params.shavex ~= 0 or params.shavey ~= 0 or params.expandx ~= 0 or params.expandy ~= 0 then
            img_shave_sizex = sizex - (params.shavex*2)
            img_shave_sizey = sizey - (params.shavey*2)
            img_expand_sizex = sizex + (params.expandx*2)
            img_expand_sizey = sizey + (params.expandy*2)
            print("--------------------------> shave & expand opticalflow [shavex/y:" .. img_shave_sizex .. "/" .. img_shave_sizey .. "] [expandx/y:" .. img_expand_sizex .. "/" .. img_expand_sizey .. "]")
            local expandcmd=string.format("%s -i %s -resize %s,%s,1,2,0,0,.5,.5,.5,.5 -resize %s,%s,1,2,0,1,.5,.5,.5,.5 -o %s 2> /var/tmp/artistic.log",gmicbin,woptical,img_shave_sizex,img_shave_sizey,img_expand_sizex,img_expand_sizey,woptical)
            -- print (expandcmd)
            os.execute(expandcmd)
        end
        -- mask opticalflow
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
        
        local fileName = build_OutFilename(params, params.start_number+math.abs(prevIndex - params.start_number),1)
        
        -- print("warping and blending image : " .. fileName)
        print("--------------------------> warping with motion flow")
        print(string.format('Reading previous file "%s".', fileName))
        imgWarped = brutewarpImage(image.load(fileName, 3), flow)
        
        -- tangentflow
        if params.dotangent then
            local tangentFileName = getFormatedFlowFileName(params.tangent_pattern, math.abs(prevIndex), math.abs(frameIdx))
            copycmd=string.format('cp %s %s',tangentFileName,wtangent)
            print(string.format('Reading tangent file "%s".',tangentFileName))
            os.execute (copycmd)
            -- resize tangent
            if params.output_size ~= 0 then
                print("--------------------------> resize tangent file")
                local resizetangentcmd=string.format("%s -i %s -resize2dx %s,5 -mul %s -o %s 2> /var/tmp/artistic.log",gmicbin,wtangent,params.output_size,motionratio,wtangent)
                -- print (resizetangentcmd)
                os.execute(resizetangentcmd)
            end
            -- expand tangent
            if params.shavex ~= 0 or params.shavey ~= 0 or params.expandx ~= 0 or params.expandy ~= 0 then
                print("--------------------------> shave & expand tangentflow [shavex/y:" .. img_shave_sizex .. "/" .. img_shave_sizey .. "] [expandx/y:" .. img_expand_sizex .. "/" .. img_expand_sizey .. "]")
                local expandcmd=string.format("%s -i %s -resize %s,%s,1,2,0,0,.5,.5,.5,.5 -resize %s,%s,1,2,0,1,.5,.5,.5,.5 -o %s 2> /var/tmp/artistic.log",gmicbin,wtangent,img_shave_sizex,img_shave_sizey,img_expand_sizex,img_expand_sizey,wtangent)
                -- print (expandcmd)
                os.execute(expandcmd)
            end
            print(string.format('Reading tangent file "%s".', wtangent))
            local mytangent = flowFile.load(wtangent)
            print("--------------------------> warping with tangent flow")
            mytangent:mul(params.tangentbooster)
            print("boost tangent : " .. params.tangentbooster)
            imgWarped = brutewarpImage(imgWarped, mytangent)
        end
        
        --gradientflow
        if params.dogradient then
            local gradientFileName = getFormatedFlowFileName(params.gradient_pattern, math.abs(prevIndex), math.abs(frameIdx))
            copycmd=string.format('cp %s %s',gradientFileName,wgradient)
            print(string.format('Reading gradient file "%s".',gradientFileName))
             -- print (copycmd)
            os.execute (copycmd)
            -- resize tangent
            if params.output_size ~= 0 then
                print("--------------------------> resize gradient file")
                -- local resizegradientcmd=string.format("flo2exr %s %s > /var/tmp/artistic.log;gmic -i %s -resize2dx %s,5 -mul %s -o %s 2> /var/tmp/artistic.log;exr2flo %s %s > /var/tmp/artistic.log" ,wgradient, wgradientexr,wgradientexr,params.output_size,motionratio,wgradientexr,wgradientexr,wgradient)
                local resizegradientcmd=string.format("%s -i %s -resize2dx %s,5 -mul %s -o %s 2> /var/tmp/artistic.log",gmicbin,wgradient,params.output_size,motionratio,wgradient)
                -- print (resizetangentcmd)
                os.execute(resizegradientcmd)
            end
            -- expand gradient
            if params.shavex ~= 0 or params.shavey ~= 0 or params.expandx ~= 0 or params.expandy ~= 0 then
                print("--------------------------> shave & expand gradientflow [shavex/y:" .. img_shave_sizex .. "/" .. img_shave_sizey .. "] [expandx/y:" .. img_expand_sizex .. "/" .. img_expand_sizey .. "]")
                local expandcmd=string.format("%s -i %s -resize %s,%s,1,2,0,0,.5,.5,.5,.5 -resize %s,%s,1,2,0,1,.5,.5,.5,.5 -o %s 2> /var/tmp/artistic.log",gmicbin,wgradient,img_shave_sizex,img_shave_sizey,img_expand_sizex,img_expand_sizey,wgradient)
                -- print (expandcmd)
                os.execute(expandcmd)
            end
            
            print(string.format('Reading gradient file "%s".', wgradient))
            local mygradient = flowFile.load(wgradient)
            print("--------------------------> warping with gradient flow")
            mygradient:mul(params.gradientbooster)
            print("boost gradient : " .. params.gradientbooster)
            imgWarped = brutewarpImage(imgWarped, mygradient)
        end
        
        --custom flow
        if params.docustom then
            local customFileName = getFormatedFlowFileName(params.custom_pattern, math.abs(prevIndex), math.abs(frameIdx))
            copycmd=string.format('cp %s %s',customFileName,wcustom)
            print(string.format('Reading custom file "%s".',customFileName))
             -- print (copycmd)
            os.execute (copycmd)
            
            -- resize custom
            if params.output_size ~= 0 then
                print("--------------------------> resize custom file")
                -- local resizecustomcmd=string.format("flo2exr %s %s > /var/tmp/artistic.log;gmic -i %s -resize2dx %s,5 -mul %s -o %s 2> /var/tmp/artistic.log;exr2flo %s %s > /var/tmp/artistic.log" ,wcustom, wcustomexr,wcustomexr,params.output_size,motionratio,wcustomexr,wcustomexr,wcustom)
                local resizecustomcmd=string.format("%s -i %s -resize2dx %s,5 -mul %s -o %s 2> /var/tmp/artistic.log",gmicbin,wcustom,params.output_size,motionratio,wcustom)
                -- print (resizecustomcmd)
                os.execute(resizecustomcmd)
            end
            -- expand custom
            if params.shavex ~= 0 or params.shavey ~= 0 or params.expandx ~= 0 or params.expandy ~= 0 then
                print("--------------------------> shave & expand customflow [shavex/y:" .. img_shave_sizex .. "/" .. img_shave_sizey .. "] [expandx/y:" .. img_expand_sizex .. "/" .. img_expand_sizey .. "]")
                -- local expandcmd=string.format("flo2exr %s %s;gmic -i %s -resize %s,%s,1,3,0,0,.5,.5,.5,.5 -resize %s,%s,1,3,0,1,.5,.5,.5,.5 -o %s 2> /var/tmp/artistic.log;exr2flo %s %s" ,wcustom, wcustomexr,wcustomexr,img_shave_sizex,img_shave_sizey,img_expand_sizex,img_expand_sizey,wcustomexr,wcustomexr,wcustom)
                local expandcmd=string.format("%s -i %s -resize %s,%s,1,2,0,0,.5,.5,.5,.5 -resize %s,%s,1,2,0,1,.5,.5,.5,.5 -o %s 2> /var/tmp/artistic.log",gmicbin,wcustom,img_shave_sizex,img_shave_sizey,img_expand_sizex,img_expand_sizey,wcustom)
                -- print (expandcmd)
                os.execute(expandcmd)
            end
            print(string.format('Reading custom file "%s".', wcustom))
            local mycustom = flowFile.load(wcustom)
            print("--------------------------> warping with custom flow")
            mycustom:mul(params.custombooster)
            print("boost custom : " .. params.custombooster)
            imgWarped = brutewarpImage(imgWarped, mycustom)
        end
        
        -- re-blend with content 
        imgWarped:mul(1-params.content_blend)
        preprocess_image = image.load(wcolor, 3)
        blend_content_image = preprocess_image:mul(params.content_blend)
        imgWarped:add(blend_content_image)
        
        -- save prewarped image
        wprewarp=string.format('%s/prewarp.png',workingdir)
        image.save(wprewarp, imgWarped)
        
        -- Read flow weights
        local weightsFileName = getFormatedFlowFileName(params.flowWeight_pattern , math.abs(frameIdx - 1) , math.abs(frameIdx))
        print(string.format('Reading consistency map "%s".', weightsFileName))
        copycmd=string.format('cp %s %s',weightsFileName,wreliable)
         -- print (copycmd)
        os.execute (copycmd)
        
        -- resize reliable
        if params.output_size ~= 0 then
            print("--------------------------> resize consistency map")
            local resizereliablecmd=string.format("%s -i %s -resize2dx %s,5 -c 0,255 -o %s 2> /var/tmp/artistic.log",gmicbin,wreliable,params.output_size,wreliable)
            -- local resizereliablecmd=string.format("%s -i %s -resize %s,%s,1,3,5 -c 0,255 -o %s 2> /var/tmp/artistic.log",gmicbin,wreliable,params.output_size,round(params.output_size/content_ratio),wreliable)
            -- print (resizereliablecmd)
            os.execute(resizereliablecmd)
        end
        
        -- expand reliable
        if params.shavex ~= 0 or params.shavey ~= 0 or params.expandx ~= 0 or params.expandy ~= 0 then
            print("--------------------------> shave & expand consistency map [shavex/y:" .. img_shave_sizex .. "/" .. img_shave_sizey .. "] [expandx/y:" .. img_expand_sizex .. "/" .. img_expand_sizey .. "]")
            --local expandcmd=string.format("gmic -i %s -div 255 -oneminus -resize %s,%s,1,1,0,0,.5,.5,.5,.5 -resize %s,%s,1,1,0,0,.5,.5,.5,.5 -oneminus -mul 255 -o %s 2> /var/tmp/artistic.log",wreliable,img_shave_sizex,img_shave_sizey,img_expand_sizex,img_expand_sizey,wreliable)
            local expandcmd=string.format("%s -i %s -resize %s,%s,1,1,0,0,.5,.5,.5,.5 -resize %s,%s,1,1,0,1,.5,.5,.5,.5 -o %s 2> /var/tmp/artistic.log",gmicbin,wreliable,img_shave_sizex,img_shave_sizey,img_expand_sizex,img_expand_sizey,wreliable)
            -- print (expandcmd)
            os.execute(expandcmd)
        end
        
    print(string.format('Reading consistency map "%s".', wreliable))
    flowWeight = image.load(wreliable):float()
    return imgWarped,flowWeight
end
