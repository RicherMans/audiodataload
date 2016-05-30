local adl = require 'audiodataload._base'
local Hdf5iterator = torch.class('adl.Hdf5iterator', 'adl.BaseDataloader',adl)

local hdf5 = require 'hdf5'

local initcheck = argcheck{
    pack=true,
    {
        name='module',
        type='adl.BaseDataloader',
        help="The module to wrap around",
    },
    {
        name='filepath',
        type='string',
        help="Cachefile for the HDF5 dumped file. If this file does not exist, we dump the modules content into it",
    },
    {
        name='chunksize',
        type='number',
        help="If newfile is provided, this arguments specifies the average chunksize for a deflated file, thus also enabling deflation.",
        opt=true
    }

}

function Hdf5iterator:__init(...)
    local args = initcheck(...)
    for k,v in pairs(args) do self[k] = v end
    print("\n",self.module)

    -- Copy the wrapped modules members
    for k,v in pairs(self.module) do
        self[k]=v
    end
    assert(self.filepath ~= nil, "Filepath needs to be specified")
    -- Filepath is not found thus dump the modules content
    if not paths.filep(self.filepath) then
        -- Dumping filecontent first into hdf5
        local uttiterator = self.module:uttiterator()
        -- Hijack the module, to force it to iterate over the whole utterance
        local maxseqlength = self.module.usemaxseqlength or false
        self.module.usemaxseqlength = true
        local hdf5write = hdf5.open(self.filepath,'w')

        print("\n",self.module)

        local hdf5options = hdf5.DataSetOptions()

        if self.chunksize then
            hdf5options:setDeflate()
            hdf5options:setChunked(self.chunksize)
        end
        for done,finish,input,_,utterancepath in uttiterator do
            -- Dump into single dimensional array
            hdf5write:write(readfilelabel(utterancepath[1]),input:view(input:nElement()),hdf5options)
        end
        -- reset the hijack
        self.module.usemaxseqlength = maxseqlength
        hdf5write:close()
    end


end

-- Randomizes the input
function Hdf5iterator:random()
    self.module:random()
end

-- Number of utterances in the dataset
-- Just wrap it around the moduel
function Hdf5iterator:usize()
    return self.module:usize()
end

-- Returns the sample size of the dataset
function Hdf5iterator:size()
    return self.module:size()
end

-- Returns the data dimensions
function Hdf5iterator:dim()
    return self.module:dim()
end

function Hdf5iterator:nClasses()
    return self.module:nClasses()
end

-- Attach the opencache module to the wrapped class
function Hdf5iterator:beforeIter(...)
    self._opencache = hdf5.open(self.filepath,'r')
end

function Hdf5iterator:getSample(labels,  ids, ...)
    return self.module.getSample(self,labels,  ids, ...)
end

function Hdf5iterator:getUtterance(start,stop, ...)
    return self.module.getUtterance(self,start,stop, ...)
end

function Hdf5iterator:afterIter(...)
    self._opencache:close()
    self._opencache = nil
end

-- Passing the opened hdf5file as the first arg in the dots
function Hdf5iterator:loadAudioSample(audiofilepath,start,stop,...)
    local readfile = self._opencache:read(audiofilepath)
    local audiosize = readfile:dataspaceSize()[1]
    if stop > audiosize then
        print("Here",start,stop,audiosize)
        -- The maximum size of fitting utterances so that no sequence will be mixed with nonzeros and zeros
        audiosize = floor(audiosize/self:dim()) * self:dim()
        -- Try to fit only the seqlen utterances in a whole in
        audiosize = floor(audiosize/self.seqlen) * self.seqlen
        local bufsize = stop-start + 1
        self._buf = self._buf or torch.Tensor()
        self._buf:resize(bufsize):zero()
        if self.padding == 'left' then
            self._buf:sub(stop-audiosize+1,bufsize):copy(readfile:partial({1,audiosize}))
        else
            self._buf:sub(1,audiosize):copy(readfile:partial({1,audiosize}))
        end
        return self._buf
    else
        return readfile:partial({start,stop})
    end
end

function Hdf5iterator:loadAudioUtterance(audiofilepath,...)
    return self._opencache:read(readfilelabel(audiofilepath)):all()
end
