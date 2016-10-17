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

local profi = require 'ProFi'
local tester = torch.Tester()

local filelist = arg[1]

function modeltester:init()
    local filepath = filelist
    local dataloader = audioload.HtkDataloader(filepath)
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
    -- run the size estimation
    local _ = dataloader:sampleiterator()

    local tic = torch.tic()
    local labcount = torch.zeros(dataloader:nClasses()):zero()
    local dataset = torch.Tensor(dataloader:size(),dataloader:dim())
    local targets = torch.LongTensor(dataloader:size(),1)
    local bs = 128
    for s,e,inp,lab in dataloader:sampleiterator(bs,nil) do
        local addone = torch.Tensor(lab:size(1)):fill(1)
        labcount:indexAdd(1,lab:long(),addone)
        local bsize = inp:size(1)
        dataset[{{s-bsize+1,s}}]:copy(inp)
        targets[{{s-bsize+1,s}}]:copy(lab)
        tester:assert(inp:size(1) == lab:size(1))
    end

    tester:assert(labcount:sum() == dataloader:size())

    -- Emulate some 3 iterations over the data
    for i=1,3 do
        local tmplabcount = torch.zeros(dataloader:nClasses())
        local notrandomized = 0
        local tmptargets = torch.Tensor(dataloader:size()):zero()
        local testunique = torch.Tensor(dataloader:size()):zero()
        dataloader:shuffle()
        for s,e,inp,lab in dataloader:sampleiterator(bs,nil) do
            local bsize = inp:size(1)

            for j=1,dataset:size(1) do
                local origtarget = targets[j][1]
                for k=1,inp:size(1) do
                    if dataset[j]:equal(inp[k]) then
                        local targetid = s-bsize + k
                        testunique[targetid] = testunique[targetid]+ 1
                        tmptargets[j] = targetid
                        tester:assert(lab[k] ==origtarget)
                        break
                    end
                end
            end
            local addone = torch.Tensor(lab:size(1)):fill(1)
            tmplabcount:indexAdd(1,lab:long(),addone)
            if dataset[{{s-bsize+1,s}}]:equal(inp) then
                notrandomized = notrandomized + 1
            end

            tester:assert(inp:size(1) == lab:size(1))
        end
        local sorted,_ = tmptargets:sort()

        tester:assert(notrandomized == 0,"Randomization factor is a bit small ("..notrandomized..")")
        tester:eq(testunique,torch.ones(dataloader:size()))
        tester:eq(tmplabcount,labcount,"Labels are not the same in iteration "..i)
    end

end

function modeltester:testsize()
    local dataloader = audioload.HtkDataloader{path=filelist}
    -- Simulate 3 iterations over the dataset
    for i=1,3 do
        local numsamples = 0
        dataloader:shuffle()
        for s,e,i in dataloader:sampleiterator(2048,nil) do
            numsamples = numsamples + i:size(1)
            collectgarbage()
        end
        local size = dataloader:size()
        tester:assert(size == numsamples)
    end
end

function modeltester:benchmark()
    local timer = torch.Timer()
    local dataloader = audioload.HtkDataloader{path=filelist}
    local bsizes = {64,128,256}
    local time = 0
    profi:start()
    for s,e,inp,lab in dataloader:uttiterator() do
    end
    profi:stop()
    profi:writeReport("Report_htkuttiter.txt")
    print(" ")
    for k,bs in pairs(bsizes) do
        for i=1,3 do
            dataloader:shuffle()
            collectgarbage()
            tic = torch.tic()
            for s,e,inp,lab in dataloader:sampleiterator(bs) do
                print(string.format("Sample [%i/%i]: Time for dataloading %.4f",s,e,timer:time().real))
                timer:reset()
            end
            time = time + torch.toc(tic)
        end
        print("HTKdata: Bsize",bs,"time:",time)
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
        for s,e,k,v in dataloader:sampleiterator(batches[i],nil)do

        end
    end
end


if not filelist or filelist == "" then
    print("Please pass a filelist as first argument")
    return
end

tester:add(modeltester)
tester:run()
