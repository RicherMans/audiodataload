local adl = require 'audiodataload._base'
local Cacheiterator = torch.class('adl.Cacheiterator', 'adl.BaseDataloader', adl)

local initcheck = argcheck{
    pack=true,
    {
        name='wrappedmodule',
        type='adl.BaseDataloader',
        help="The module to wrap around",
    },
    {
    	name='cachesize',
    	type='number',
    	help='The number of samples stored in the cache. By default stores everything',
    	default=-1,
		check = function(num)
			if num > 0 then return true else return false end
		end    	
	}
}


function Cacheiterator:__init(...)
	local args = initcheck(...)
	for k,v in pairs(args) do
		self[k] = v
	end
	for k,v in pairs(self.wrappedmodule) do
		self[k] = v
	end
	-- If default == -1, then use the wrapped module's size
	self._usefullsize = false
	if self.cachesize == -1 then
		self.cachesize = self.wrappedmodule:size()
		self._usefullsize = true
	end
	-- In case a too large cachesize was passed
	if self.cachesize > self.wrappedmodule:size() then
		self.cachesize = self.wrappedmodule:size()
		self._usefullsize = true
	end
	self._cache = torch.Tensor(self.cachesize,self.wrappedmodule:dim()):fill(-1)
end


function Cacheiterator:getSample(labels,ids,...)
	-- Pass the self as the wrapped modules self, since it contains the 
	-- Sampletofeat and sampletoclassrange variables
	if self._usefullsize and self._cache:index(1,ids):eq(-1):all() then
		local samples = self.wrappedmodule.getSample(self,labels,ids, ...)
		-- Store in cache
		self._cache:indexCopy(1,ids,samples)
		return samples
	elseif not self._usefullsize then
		-- Get the subset of the indices which are in range 
		local subids = ids:maskedSelect(ids:le(self.cachesize))
		--It can happen that subids consists or not a single value 
		if subids:dim() > 0 and self._cache:index(1,subids):eq(-1):all() then
			local samples = self.wrappedmodule.getSample(self,labels,ids, ...)
			-- In case of cachesize is set to a larger value than the cache
			if samples:size(1) > self.cachesize then
				-- In case cache has not the same size as the batch
				self._cache:index(1,subids):copy(samples:index(1,subids))
			-- Insert into the cache at the indexes the (sorted) samples subset
			else
				self._cache:index(1,subids):copy(samples[{{1,subids:size(1)}}])
			end

			return samples
		-- All the values are stored in the cache, just obtain them
		elseif subids:dim() > 0 then
			local otherids = ids:maskedSelect(ids:gt(self.cachesize))
			local otherlabels = labels:index(1,otherids)
			-- The first M samples
			local cachesamples = self._cache:index(1,subids)
			-- The other N samples, so that it adds up to M+N
			local memorysamples = self.wrappedmodule.getSample(self,otherlabels,otherids, ...)
			return torch.cat(cachesamples,memorysamples,1)
		-- We did not find the data within the cache at all, so just load it and return
		else
			return self.wrappedmodule.getSample(self,labels,ids, ...)
		end

		
	end

	return self._cache:index(1,ids)
end

function Cacheiterator:loadAudioSample(labels,ids,...)
	return self.wrappedmodule:loadAudioSample(labels,ids, ...)
end


-- Number of utterances in the dataset
-- Just wrap it around the moduel
function Cacheiterator:usize()
    return self.wrappedmodule:usize()
end

-- Returns the sample size of the dataset
function Cacheiterator:size()
    return self.wrappedmodule:size()
end

-- Returns the data dimensions
function Cacheiterator:dim()
    return self.wrappedmodule:dim()
end

function Cacheiterator:nClasses()
    return self.wrappedmodule:nClasses()
end
