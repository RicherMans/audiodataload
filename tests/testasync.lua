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
    local asyncdata = audioload.Asynciterator(dataloader,3)

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


function modeltester:randomizedtest()
    local filepath = htkfilelist
    local dataloader = audioload.HtkDataloader(filepath)
    local asynciter = audioload.Asynciterator(dataloader,4)

    local valuetolab = {}
    local numvalues = 0
    local tic = torch.tic()
    local valuetoidx = {}
    local starttovalue = {}
    local labcount = torch.zeros(asynciter:nClasses())
    print("Init sampleiterator")
    for s,e,inp,lab in asynciter:sampleiterator(1,nil,true) do
        valuetolab[inp[1][1]] = lab[1]
        valuetoidx[inp[1][1]] = s
        labcount[lab[1]] = labcount[lab[1]] + 1
        starttovalue[s] = inp[1][1]
        numvalues = numvalues + 1
        tester:assert(inp:size(1) == lab:size(1))
    end

    local numsamples = asynciter:size()

    tester:assert(labcount:sum() == numsamples)
    tester:assert(numvalues == numsamples)
    local maxrandomizeerr = math.ceil(numsamples/100)


    -- Emulate some 10 iterations over the data
    print("Running 5 iterations ")
    for i=1,5 do
        local tmpnumvalues = numvalues
        local randomizecount = 0
        local tmpvaluetolab = shallowcopy(valuetolab)
        local tmpvaluetoidx = {}
        local itercount = 0
        local startpoints = {}
        local tmplabcount = labcount:clone()
        for i=1,numsamples do
            startpoints[#startpoints+1] = i
        end
        for s,e,inp,lab in asynciter:sampleiterator(1,nil,true) do
            itercount = itercount + 1
            tmpvaluetoidx[inp[1][1]] = s
            if valuetolab[inp[1][1]] and valuetolab[inp[1][1]] == lab[1] then
                tmpnumvalues = tmpnumvalues - 1
                tmpvaluetolab[inp[1][1]] = nil
            end
            tmplabcount[lab[1]] = tmplabcount[lab[1]] -1
            startpoints[s] = nil
            -- Count of non sucessfull randomized values ( can happen )
            if starttovalue[s] == inp[1][1] then
                randomizecount = randomizecount + 1
            end
            tester:assert(inp:size(1) == lab:size(1))
        end
        print("Iteration "..i.." done")  
        tester:eq(tmplabcount,torch.zeros(asynciter:nClasses()),"Labels are not the same in iteration "..i)
        
        -- pick 4 as a valid number of unchanged indices
        tester:assert(next(startpoints) == nil)
        -- Check if the randomization works, we use some artificial threshold. Not that important if that fails by some larger count
        tester:assert(randomizecount < maxrandomizeerr,"Error in iteration "..i..", randomize count is "..randomizecount)
        tester:assert(tmpnumvalues == 0,"Error in iteration "..i.." leftover "..tmpnumvalues)
    end
end

tester:add(modeltester)
tester:run()