require 'torch'
require 'nn'
require 'xlua'

-- Add to cpath
local info = debug.getinfo(1,'S')
local script_path=info.source:sub(2):gsub("testasync",'?')
package.path = package.path .. ";".. script_path .. ";../".. script_path..  ";../../" ..script_path
local modeltester = torch.TestSuite()
-- Inits the dataloaders
local audioload = paths.dofile('../init.lua')

local tester = torch.Tester()
ProFi = require 'ProFi'

local initcheck  = argcheck{
    {
        name='wavedatafile',
        type='string',
        help="Wavefile to run experiments with",
    },
    {
        name='htkdatafile',
        type='string',
        help='HTK filelist'
    }
}

local wavefilelist,htkfilelist = initcheck(arg[1],arg[2])

function shallowcopy(orig)
    local orig_type = type(orig)
    local copy
    if orig_type == 'table' then
        copy = {}
        for orig_key, orig_value in pairs(orig) do
            copy[orig_key] = orig_value
        end
    else -- number, string, boolean, etc
        copy = orig
    end
    return copy
end


function modeltester:init()
    local filepath = wavefilelist
    local dataloader = audioload.WaveDataloader(filepath,100)
    local asyncdata = audioload.Asynciterator(dataloader)
end


function modeltester:testsampleiteratormultipleWAV()
    local filepath = wavefilelist
    local dataloader = audioload.WaveDataloader(filepath,400)
    local asyncdata = audioload.Asynciterator(dataloader,3)

    for s,e,inp,lab in dataloader:sampleiterator(128) do
        tester:assert(inp:size(1) == lab:size(1))
    end

    -- Multiple loops
    for s,e,inp,lab in asyncdata:sampleiterator(44) do
        tester:assert(inp:size(1) == lab:size(1))
    end

    for s,e,inp,lab in asyncdata:sampleiterator(44) do
        tester:assert(inp:size(1) == lab:size(1))
    end
end

function modeltester:testsampleiteratormultipleHTK()
    local filepath = htkfilelist
    local dataloader = audioload.HtkDataloader(filepath)
    local asyncdata = audioload.Asynciterator(dataloader,1)

    local classsizes= torch.Tensor(asyncdata:nClasses()):zero()
    for s,e,inp,lab in asyncdata:sampleiterator(128) do

        tester:assert(inp:size(1) == lab:size(1))
        local addone = torch.Tensor(lab:size(1)):fill(1)
        classsizes:indexAdd(1,lab:long(),addone)
    end
    tester:assert(classsizes:sum()==asyncdata:size())

    -- Multiple iterations over data
    for i=1,5 do
        local tmpclasssizes = torch.Tensor(asyncdata:nClasses()):zero()
        for s,e,inp,lab in asyncdata:sampleiterator(44,nil,true) do
            local addone = torch.Tensor(lab:size(1)):fill(1)
            tmpclasssizes:indexAdd(1,lab:long(),addone)
            tester:assert(inp:size(1) == lab:size(1))
        end
        tester:eq(tmpclasssizes,classsizes)
    end
end

function modeltester:testsampleiteratorsinglethread()
    local filepath = wavefilelist
    local dataloader = audioload.WaveDataloader(filepath,100)
    local asyncdata = audioload.Asynciterator(dataloader,1)

    for s,e,inp,lab in asyncdata:sampleiterator(45) do
        tester:assert(inp:size(1) == lab:size(1))
    end
end

function modeltester:benchmark()
    local filepath = wavefilelist
    local dataloader = audioload.WaveDataloader(filepath,100)
    local asyncdata = audioload.Asynciterator(dataloader,3)

    local bsizes = {1,10,128}
    for k,bs in pairs(bsizes) do
        local rawtime, asynctime = 0,0
        for i=1,3 do
            local tic = torch.tic()
            for s,e,inp,lab in asyncdata:sampleiterator(bs,nil,true) do

            end
            asynctime = asynctime + torch.toc(tic)
            collectgarbage()
            tic = torch.tic()
            for s,e,inp,lab in dataloader:sampleiterator(bs,nil,true) do

            end
            rawtime = rawtime + torch.toc(tic)
            collectgarbage()
        end
        print("Wave datafiles: For batchsize "..bs," Asynctime: "..asynctime," Wavetime: ",rawtime)
    end

    dataloader = audioload.HtkDataloader(htkfilelist)
    asyncdata = audioload.Asynciterator(dataloader,3)
    local bsizes = {1,10,128}
    for k,bs in pairs(bsizes) do
        local rawtime, asynctime = 0,0
        for i=1,3 do
            local tic = torch.tic()
            for s,e,inp,lab in asyncdata:sampleiterator(bs,nil,true) do

            end
            asynctime = asynctime + torch.toc(tic)
            collectgarbage()
            tic = torch.tic()
            for s,e,inp,lab in dataloader:sampleiterator(bs,nil,true) do

            end
            rawtime = rawtime + torch.toc(tic)
            collectgarbage()
        end
        print("HTK data: For batchsize "..bs," Asynctime: "..asynctime," Htkdataloadertime: ",rawtime)
    end
end


function modeltester:meanstdnormalizationexample()
    local dataloader = audioload.HtkDataloader(htkfilelist)
    local asynciter = audioload.Asynciterator(dataloader,4)
    local datasize = asynciter:size()
    local samplesize = 300
    -- Simulate mean sampling
    local iter = asynciter:sampleiterator(samplesize,-1,true)
    local s,e,samples = iter()
    local maxsize = math.min(datasize,samplesize)
    tester:assert(samples:size(1) == maxsize,"Sample size "..samples:size(1).." should be: "..maxsize)

    iter = asynciter:sampleiterator(samplesize,-1,true)
    local s,e,samples = iter()
    local maxsize = math.min(datasize,samplesize)
    tester:assert(samples:size(1) == maxsize,"Sample size "..samples:size(1).." should be: "..maxsize)

    for k,v,inp,lab in asynciter:sampleiterator(128,-1,true) do
        tester:assert(inp:size(1) <= 128)
    end

    for k,v,inp,lab in asynciter:sampleiterator(128,-1,true) do
        tester:assert(inp:size(1) <= 128)
    end
end


function modeltester:randomizedtest()
    local filepath = htkfilelist
    local dataloader = audioload.HtkDataloader(filepath)
    local asynciter = audioload.Asynciterator(dataloader,2)

    local valuetolab = {}
    local numvalues = 0
    local labcount = torch.zeros(asynciter:nClasses())
    local dataset = torch.FloatTensor(asynciter:size(),asynciter:dim())
    local targets = torch.LongTensor(asynciter:size(),1)
    for s,e,inp,lab in asynciter:sampleiterator(1,nil) do
        valuetolab[inp[1][1]] = lab[1]
        labcount[lab[1]] = labcount[lab[1]] + 1
        numvalues = numvalues + 1
        dataset[s]:copy(inp:view(inp:nElement()))
        targets[s]:copy(lab)
        tester:assert(inp:size(1) == lab:size(1))
    end

    local numsamples = asynciter:size()

    tester:assert(labcount:sum() == numsamples)
    tester:assert(numvalues == numsamples)
    local maxrandomizeerr = math.ceil(numsamples/100)


    -- Emulate some 10 iterations over the data
    print("Running 2 iterations ")
    for i=1,2 do
        local tmpnumvalues = numvalues
        local randomizecount = 0
        local tmpvaluetolab = shallowcopy(valuetolab)
        local itercount = 0
        local startpoints = {}
        local tmplabcount = labcount:clone()
        for i=1,numsamples do
            startpoints[#startpoints+1] = i
        end
        for s,e,inp,lab in asynciter:sampleiterator(1,nil,true) do
            itercount = itercount + 1
            if valuetolab[inp[1][1]] and valuetolab[inp[1][1]] == lab[1] then
                tmpnumvalues = tmpnumvalues - 1
                tmpvaluetolab[inp[1][1]] = nil
            end
            tmplabcount[lab[1]] = tmplabcount[lab[1]] -1
            startpoints[s] = nil
            for j=1,dataset:size(1) do
                if dataset[j]:equal(inp:view(inp:nElement())) then
                    tester:assert(targets[j]:equal(lab),"Randomized labels are not correct")
                    break
                end
            end
            -- Count of non sucessfull randomized values ( can happen )
            if dataset[s]:equal(inp:view(inp:nElement())) then
                randomizecount = randomizecount + 1
            end
            tester:assert(inp:size(1) == lab:size(1))
        end
        print("Iteration "..i.." done")  
        tester:eq(tmplabcount,torch.zeros(asynciter:nClasses()),"Labels are not the same in iteration "..i)
    
        -- Check if all indices were visited        
        tester:assert(next(startpoints) == nil)
        -- Check if the randomization works, we use some artificial threshold. Not that important if that fails by some larger count
        tester:assert(randomizecount < maxrandomizeerr,"Error in iteration "..i..", randomize count is "..randomizecount .. "/"..numsamples)
        tester:assert(tmpnumvalues == 0,"Error in iteration "..i.." leftover "..tmpnumvalues)
    end
end

tester:add(modeltester)
tester:run()