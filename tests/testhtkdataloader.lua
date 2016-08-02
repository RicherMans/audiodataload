require 'torch'
require 'nn'
require 'xlua'

-- Add to cpath
local info = debug.getinfo(1,'S')
local script_path=info.source:sub(2):gsub("testhtkdataloader",'?')
package.path = package.path .. ";".. script_path .. ";../".. script_path..  ";../../" ..script_path
local modeltester = torch.TestSuite()
-- Inits the dataloaders
local audioload = paths.dofile('../init.lua')

local tester = torch.Tester()

local filelist = arg[1]

function modeltester:init()
    local filepath = filelist
    local dataloader = audioload.HtkDataloader(filepath)
end
function modeltester:testbatchUtterance()
    local filepath = filelist
    local dataloader = audioload.HtkDataloader{path=filepath}

    local iter = dataloader:uttiterator()
    for i,k,v,t in iter do
        tester:assert(type(i) == 'number')
        tester:assert(type(k) == 'number')
        tester:assert(torch.isTensor(v))
    end
end
function modeltester:testnonrandomizedsamples()
    local filepath = filelist
    local dataloader = audioload.HtkDataloader(filepath)
    local classsizes= torch.Tensor(dataloader:nClasses()):zero()

    local batchsize = 128
    for s,e,k,v in dataloader:sampleiterator(batchsize) do
        local addone = torch.Tensor(v:size(1)):fill(1)
        classsizes:indexAdd(1,v:long(),addone)
    end
    tester:assert(classsizes:sum()==dataloader:size())
    for i=1,3 do
        local tmpclasssizes = torch.Tensor(dataloader:nClasses()):zero()
        for s,e,k,v in dataloader:sampleiterator(batchsize) do
            local addone = torch.Tensor(v:size(1)):fill(1)
            tmpclasssizes:indexAdd(1,v:long(),addone)
        end
        tester:eq(tmpclasssizes,classsizes)
    end

end

--
function modeltester:testrandomize()
    local filepath = filelist
    local dataloader = audioload.HtkDataloader{path=filepath}

    local tic = torch.tic()
    local labcount = torch.zeros(dataloader:nClasses())
    local dataset = torch.Tensor(dataloader:size(),dataloader:dim())
    local targets = torch.LongTensor(dataloader:size(),1)
    for s,e,inp,lab in dataloader:sampleiterator(1,nil) do
        labcount[lab[1]] = labcount[lab[1]] + 1
        dataset[s]:copy(inp:view(inp:nElement()))
        targets[s]:copy(lab)
        tester:assert(inp:size(1) == lab:size(1))
    end

    tester:assert(labcount:sum() == dataloader:size())

    -- Emulate some 3 iterations over the data
    for i=1,3 do
        local tmplabcount = labcount:clone()
        local notrandomized = 0
        for s,e,inp,lab in dataloader:sampleiterator(1,nil,true) do
            
            for j=1,dataset:size(1) do
                if dataset[j]:equal(inp:view(inp:nElement())) then
                    tester:assert(targets[j]:equal(lab),"Randomized labels are not correct")
                    break
                end
            end
            if dataset[s]:equal(inp:view(inp:nElement())) then
                notrandomized = notrandomized + 1
            end
            
            tmplabcount[lab[1]] = tmplabcount[lab[1]] -1
            tester:assert(inp:size(1) == lab:size(1))
        end
        tester:assert(notrandomized < 10,"Randomization factor is a bit small")
        tester:eq(tmplabcount,torch.zeros(dataloader:nClasses()),"Labels are not the same in iteration "..i)
    end
    
end

function modeltester:testsize()
    local dataloader = audioload.HtkDataloader{path=filelist}
    local size = dataloader:size()
    -- Simulate 3 iterations over the dataset
    for i=1,3 do
        local numsamples = 0
        for s,e,i in dataloader:sampleiterator(2048,nil,true) do
            numsamples = numsamples + i:size(1)
        end
        tester:assert(size == numsamples)
    end
end

function modeltester:testloadAudioSample()
    local dataloader = audioload.HtkDataloader(filelist)
    local f=assert(io.open(filelist))
    local firstline = f:read()
    local feat = firstline:split(" ")[1]
    -- print(dataloader:loadAudioSample(feat,39,40))
    -- dataloader:loadAudioSample
end

function modeltester:testusize()
    local dataloader = audioload.HtkDataloader{path=filelist}
    local size = dataloader:usize()
    -- Simulate 3 iterations over the dataset
    for i=1,3 do
        local numutts = 0
        for _ in dataloader:uttiterator() do
            numutts = numutts + 1
        end
        tester:assert(size == numutts)
    end
end

function modeltester:testdifferentbatchsize()
    local batches = {1,32,64,128,256,512,1024}
    local dataloader = audioload.HtkDataloader{path=filelist}
    -- Just test if any error happen
    for i=1,#batches do
        for s,e,k,v in dataloader:sampleiterator(batches[i],nil,true)do

        end
    end
end


if not filelist or filelist == "" then
    print("Please pass a filelist as first argument")
    return
end

tester:add(modeltester)
tester:run()
