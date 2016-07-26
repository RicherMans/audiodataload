require 'torch'
require 'nn'
require 'xlua'

-- Add to cpath
local info = debug.getinfo(1,'S')
local script_path=info.source:sub(2):gsub("testcacheiterator",'?')
package.path = package.path .. ";".. script_path .. ";../".. script_path..  ";../../" ..script_path
local modeltester = torch.TestSuite()
-- Inits the dataloaders
local audioload = paths.dofile('../init.lua')

local tester = torch.Tester()


local initcheck  = argcheck{
    {
        name='htkdatafile',
        type='string',
        help='HTK filelist'
    }
}

local htkfilelist = initcheck(arg[1])

function modeltester:init()
    local filepath = htkfilelist
    local dataloader = audioload.HtkDataloader(filepath)
    audioload.Cacheiterator(dataloader,2)
end
function modeltester:iterfull()
    local filepath = htkfilelist
    local dataloader = audioload.HtkDataloader(filepath)
    local cacheiter = audioload.Cacheiterator(dataloader)

    local valuetolab = {}
    local numvalues = 0
    local tic = torch.tic()
    for s,e,inp,lab in cacheiter:sampleiterator(1,nil,true) do
        valuetolab[inp[1][1]] = lab[1]
        numvalues = numvalues + 1
    	tester:assert(inp:size(1) == lab:size(1))
    end

    -- Emulate some 10 iterations over the data
    for i=1,10 do
        local tmpnumvalues = numvalues
        for s,e,inp,lab in cacheiter:sampleiterator(1,nil,true) do
            if valuetolab[inp[1][1]] then
                if valuetolab[inp[1][1]] == lab[1] then
                    tmpnumvalues = tmpnumvalues - 1
                end
            end
            tester:assert(inp:size(1) == lab:size(1))
        end
        tester:assert(tmpnumvalues == 0,"Error in iteration "..i)
    end
    -- Check if randomization is working (e.g. the labels of the value arrays are the same)
    print("First iter:")
    print(torch.toc(tic))
    tic = torch.tic()
    for s,e,inp,lab in cacheiter:sampleiterator(128) do
    	tester:assert(inp:size(1) == lab:size(1))
    end
    print(torch.toc(tic))
end

-- function modeltester:itersmall()
--     local filepath = htkfilelist
--     local dataloader = audioload.HtkDataloader(filepath)
--     -- Only using 5 utterances to store
--     local cacheiter = audioload.Cacheiterator(dataloader)

--     local tic = torch.tic()
--     for s,e,inp,lab in cacheiter:sampleiterator(128) do
--     	tester:assert(inp:size(1) == lab:size(1))
--     end
--     print("First iter:")
--     print(torch.toc(tic))
--     tic = torch.tic()
--     for s,e,inp,lab in cacheiter:sampleiterator(128) do
--     	tester:assert(inp:size(1) == lab:size(1))
--     end
--     print(torch.toc(tic))
-- end

tester:add(modeltester)
tester:run()
