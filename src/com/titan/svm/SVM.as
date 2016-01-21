package com.titan.svm 
{
	/**
	 * ...
	 * @author DainSiahTill
	 */
	public class SVM 
	{
		public var data:Array;
		public var labels:Vector.<int>;
		public var options:Object;
		
		public var kernelType:String;
		public var rbfSigma:Number;
		public var kernel:Function;
		public var N:int;
		public var D:int;
		
		public var alpha:Array;
		public var b:Number;
		public var usew_:Boolean = false;
		public var kernelResults:Array;
		public var w:Array;
		public var ai:int;
		public var aj:int;
		
		public function SVM() 
		{
			
		}
		
		public function train(data:Array, labels:Vector.<int>, options:Object):Object 
		{
			this.data = data;
			this.labels = labels;
			this.options = options || {};
			
			var C:Number = options.C || 1.0; // C value. Decrease for more regularization
			var tol:Number = options.tol || 1e-4; // numerical tolerance. Don't touch unless you're pro
			var alphatol:Number = options.alphatol || 1e-7; // non-support vectors for space and time efficiency are truncated. To guarantee correct result set this to 0 to do no truncating. If you want to increase efficiency, experiment with setting this little higher, up to maybe 1e-4 or so.
			var maxiter:int = options.maxiter || 10000; // max number of iterations
			var numpasses:int = options.numpasses || 10; // how many passes over data with no change before we halt? Increase for more precision.

			// instantiate kernel according to options. kernel can be given as string or as a custom function
			var kernel:Function = linearKernel;
			this.kernelType = "linear";
			if ("kernel" in options) 
			{
				if (typeof options.kernel === "string") 
				{
				  // kernel was specified as a string. Handle these special cases appropriately
				  if (options.kernel === "linear") 
				  { 
					this.kernelType = "linear"; 
					kernel = linearKernel; 
				  }
				  if (options.kernel === "rbf") 
				  { 
					var rbfSigma:Number = options.rbfsigma || 0.5;
					this.rbfSigma = rbfSigma; // back this up
					this.kernelType = "rbf";
					kernel = makeRbfKernel(rbfSigma);
				  }
				} 
				else 
				{
				  // assume kernel was specified as a function. Let's just use it
				  this.kernelType = "custom";
				  kernel = options.kernel;
				}
			}

			// initializations
			this.kernel = kernel;
			this.N = data.length;
			var N:int = this.N;
			this.D = data[0].length;
			var D:int = this.D;
			this.alpha = zeros(N);
			this.b = 0.0;
			this.usew_ = false; // internal efficiency flag

			// Cache kernel computations to avoid expensive recomputation.
			// This could use too much memory if N is large.
			if (options.memoize) 
			{
				this.kernelResults = new Array(N);
				for (var i:int = 0; i < N; i++) 
				{
				  this.kernelResults[i] = new Array(N);
				  for (var j:int = 0; j < N; j++) 
				  {
					this.kernelResults[i][j] = kernel(data[i],data[j]);
				  }
				}
			}

			// run SMO algorithm
			var iter:int = 0;
			var passes:int = 0;
			while (passes < numpasses && iter < maxiter) 
			{
				var alphaChanged:int = 0;
				for (i = 0; i < N; i++) 
				{
				  var Ei:Number = this.marginOne(data[i]) - labels[i];
				  if((labels[i]*Ei < -tol && this.alpha[i] < C)
				   || (labels[i] * Ei > tol && this.alpha[i] > 0))
				   {
					// alpha_i needs updating! Pick a j to update it with
					j = i;
					while (j === i)
					{
						j = randi(0, this.N);
					}
					var Ej:Number = this.marginOne(data[j]) - labels[j];
					
					// calculate L and H bounds for j to ensure we're in [0 C]x[0 C] box
					ai= this.alpha[i];
					aj= this.alpha[j];
					var L:Number = 0;
					var H:Number = C;
					if (labels[i] === labels[j]) 
					{
					  L = Math.max(0, ai + aj - C);
					  H = Math.min(C, ai + aj);
					} 
					else 
					{
					  L = Math.max(0, aj - ai);
					  H = Math.min(C, C + aj - ai);
					}
					
					if (Math.abs(L - H) < 1e-4) 
					{
						continue;
					}

					var eta:Number = 2 * this.kernelResult(i, j) - this.kernelResult(i, i) - this.kernelResult(j, j);
					if (eta >= 0)
					{
						continue;
					}
					
					// compute new alpha_j and clip it inside [0 C]x[0 C] box
					// then compute alpha_i based on it.
					var newaj:Number = aj - labels[j] * (Ei - Ej) / eta;
					if (newaj > H) 
					{
						newaj = H;
					}
					if (newaj < L) 
					{
						newaj = L;
					}
					if (Math.abs(aj - newaj) < 1e-4)
					{
						continue; 
					}
					this.alpha[j] = newaj;
					var newai:Number = ai + labels[i] * labels[j] * (aj - newaj);
					this.alpha[i] = newai;
					
					// update the bias term
					var b1:Number = this.b - Ei - labels[i] * (newai - ai) * this.kernelResult(i, i)
							 - labels[j] * (newaj - aj) * this.kernelResult(i, j);
					var b2:Number = this.b - Ej - labels[i] * (newai - ai) * this.kernelResult(i, j)
							 - labels[j] * (newaj - aj) * this.kernelResult(j, j);
					this.b = 0.5 * (b1 + b2);
					
					if (newai > 0 && newai < C) 
					{
						this.b = b1;
					}
					if (newaj > 0 && newaj < C)
					{
						this.b = b2;
					}
					
					alphaChanged++;
					
				  } // end alpha_i needed updating
				} // end for i=1..N
				iter++;
				//console.log("iter number %d, alphaChanged = %d", iter, alphaChanged);
				if (alphaChanged == 0)
				{
					passes++;
				}
				else
				{
					passes= 0;
				}
			} // end outer loop
			// if the user was using a linear kernel, lets also compute and store the
			// weights. This will speed up evaluations during testing time
			if (this.kernelType === "linear") 
			{
				// compute weights and store them
				this.w = new Array(this.D);
				for (j = 0; j < this.D; j++) 
				{
				  var s:Number = 0.0;
				  for (i = 0; i < this.N; i++) 
				  {
					s += this.alpha[i] * labels[i] * data[i][j];
				  }
				  this.w[j] = s;
				  this.usew_ = true;
				}
			} 
			else
			{
				// okay, we need to retain all the support vectors in the training data,
				// we can't just get away with computing the weights and throwing it out
				// But! We only need to store the support vectors for evaluation of testing
				// instances. So filter here based on this.alpha[i]. The training data
				// for which this.alpha[i] = 0 is irrelevant for future. 
				var newdata:Array = [];
				var newlabels:Vector.<int> = new Vector.<int>();
				var newalpha:Array = [];
				for (i = 0; i < this.N; i++) 
				{
				  //console.log("alpha=%f", this.alpha[i]);
				  if (this.alpha[i] > alphatol) 
				  {
					newdata.push(this.data[i]);
					newlabels.push(this.labels[i]);
					newalpha.push(this.alpha[i]);
				  }
				}
				
				// store data and labels
				this.data = newdata;
				this.labels = newlabels;
				this.alpha = newalpha;
				this.N = this.data.length;
				//console.log("filtered training data from %d to %d support vectors.", data.length, this.data.length);
			}
			var trainstats:Object = {};
			trainstats.iters= iter;
			return trainstats;
		}
		
		public function marginOne(inst:Object):Number
		{
		  var f:Number = this.b;
		  // if the linear kernel was used and w was computed and stored,
		  // (i.e. the svm has fully finished training)
		  // the internal class variable usew_ will be set to true.
		  if (this.usew_) 
		  {
			// we can speed this up a lot by using the computed weights
			// we computed these during train(). This is significantly faster
			// than the version below
			for (var j:int = 0; j < this.D; j++) 
			{
			  f += inst[j] * this.w[j];
			}
		  } 
		  else 
		  {
			for (var i:int = 0; i < this.N; i++) 
			{
			  f += this.alpha[i] * this.labels[i] * this.kernel(inst, this.data[i]);
			}
		  }
		  return f;
		}
		
		public function predictOne(inst:Object):int
		{ 
		  return this.marginOne(inst) > 0 ? 1 : -1; 
		}
		
		public function margins(data:Array):Array
		{
		  // go over support vectors and accumulate the prediction. 
		  var N:int = data.length;
		  var margins:Array = new Array(N);
		  for (var i:int = 0; i < N; i++) 
		  {
			margins[i] = this.marginOne(data[i]);
		  }
		  return margins;
		}

		public function kernelResult(i:int, j:int):Number
		{
		  if (this.kernelResults) 
		  {
			return this.kernelResults[i][j];
		  }
		  return this.kernel(this.data[i], this.data[j]);
		}

		// data is NxD array. Returns array of 1 or -1, predictions
		public function predict(data:Array):Array
		{
		  var margs:Array = this.margins(data);
		  for (var i:int = 0; i < margs.length; i++) 
		  {
			margs[i] = margs[i] > 0 ? 1 : -1;
		  }
		  return margs;
		}
		
		public function getWeights():Object
		{
		  // DEPRECATED
		  var w:Array = new Array(this.D);
		  for (var j:int = 0; j < this.D; j++) 
		  {
			var s:Number = 0.0;
			for (var i:int = 0; i < this.N; i++) 
			{
			  s += this.alpha[i] * this.labels[i] * this.data[i][j];
			}
			w[j]= s;
		  }
		  return {w: w, b: this.b};
		}
		
		// Kernels
		public function makeRbfKernel(sigma:Number):Function
		{
			return function(v1:Array, v2:Array):Number
			{
				var s:Number = 0;
				for (var q:int = 0; q < v1.length; q++) 
				{
				  s += (v1[q] - v2[q]) * (v1[q] - v2[q]);
				}
				return Math.exp( -s / (2.0 * sigma * sigma));
			}
		}

		public function linearKernel(v1:Array, v2:Array):Number
		{
			var s:Number = 0;
			for (var q:int = 0; q < v1.length; q++) 
			{
				s += v1[q] * v2[q];	
			}
			return s;
		}

		// Misc utility functions
		// generate random floating point number between a and b
		public function randf(a:Number, b:Number):Number
		{
			return Math.random() * (b - a) + a;
		}

		// generate random integer between a and b (b excluded)
		public function randi(a:Number, b:Number):Number
		{
			return Math.floor(Math.random() * (b - a) + a);
		}

		// create vector of zeros of length n
		public function zeros(n:int):Array
		{
			var arr:Array = new Array(n);
			for (var i:int = 0; i < n; i++) 
			{
				arr[i] = 0;
			}
			return arr;
		}
	}

}