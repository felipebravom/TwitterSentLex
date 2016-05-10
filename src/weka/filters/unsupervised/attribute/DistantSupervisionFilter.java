package weka.filters.unsupervised.attribute;

import weka.core.Instances;
import weka.filters.SimpleBatchFilter;

public abstract class DistantSupervisionFilter extends SimpleBatchFilter{

	
	private static final long serialVersionUID = 1L;
	

	public abstract Instances mapTargetInstance(Instances inp);

	public String getUsefulInfo(){
		return("");
	}

}
