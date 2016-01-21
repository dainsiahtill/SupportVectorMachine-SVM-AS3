package
{
	import com.titan.svm.SVM;
	import flash.display.Sprite;
	import flash.events.Event;
	
	/**
	 * ...
	 * @author DainSiahTill
	 */
	public class Main extends Sprite 
	{
		public static const WIDTH:int = 800;
		public static const HEIGHT:int = 600;
		
		public var N:int = 10; //number of data points
		public var data:Array = new Array(N);
		public var labels:Vector.<int> = new Vector.<int>(N);
		public var wb:Object; // weights and offset structure
		public var ss:Number= 50.0; // scaling factor for drawing
		public var svm:SVM = new SVM();
		public var trainstats:Object;
		public var dirty:Boolean = true;
		public var kernelid:int = 1;
		public var rbfKernelSigma:Number = 0.5;
		public var svmC:Number = 1.0;

		public function Main() 
		{
			if (stage) init();
			else addEventListener(Event.ADDED_TO_STAGE, init);
		}
		
		private function init(e:Event = null):void 
		{
			removeEventListener(Event.ADDED_TO_STAGE, init);
			// entry point
			data[0]=[-0.4326  ,  1.1909 ];
			data[1]= [3.0, 4.0];
			data[2]=[0.1253 , -0.0376   ];
			data[3]=[0.2877 ,   0.3273  ];
			data[4]=[-1.1465 ,   0.1746 ];
			data[5]=[1.8133 ,   2.1139  ];
			data[6]=[2.7258 ,   3.0668  ];
			data[7]=[1.4117 ,   2.0593  ];
			data[8]=[4.1832 ,   1.9044  ];
			data[9]=[1.8636 ,   1.1677  ];
			labels[0]= 1;
			labels[1]= 1;
			labels[2]= 1;
			labels[3]= 1;
			labels[4]= 1;
			labels[5]= -1;
			labels[6]= -1;
			labels[7]= -1;
			labels[8]= -1;
			labels[9]= -1;

			retrainSVM();
			
			addEventListener(Event.ENTER_FRAME, onEnterFrame);
		}
		
		private function onEnterFrame(e:Event):void 
		{
			draw();
		}
		
		public function retrainSVM():void
		{
		  if (kernelid === 1) 
		  {
			trainstats= svm.train(data, labels, { kernel: 'rbf', rbfsigma: rbfKernelSigma, C: svmC});
		  }
		  if (kernelid === 0) 
		  {
			trainstats= svm.train(data, labels, { kernel: 'linear' , C: svmC});
			wb = svm.getWeights();
		  }
		  
		  dirty = true; // to redraw screen
		}
		
		public function draw():void 
		{
			graphics.clear();
			// draw decisions in the grid
			var density:Number = 4.0;
			for (var x:Number = 0.0; x <= WIDTH; x += density) 
			{
			  for (var y:Number = 0.0; y <= HEIGHT; y += density) 
			  {
				var dec:Number = svm.marginOne([(x - WIDTH / 2) / ss, (y - HEIGHT / 2) / ss]);
				if (dec > 0)
				{
					graphics.beginFill(0x4DECC0);
				}
				else 
				{
					graphics.beginFill(0xE450E9);
				}
				graphics.drawRect(x - density / 2 - 1, y - density - 1, density + 2, density + 2);
			  }
			}
			
			// draw axes
			graphics.lineStyle(2, 0xcccccc);
			graphics.moveTo(0, HEIGHT/2);
			graphics.lineTo(WIDTH, HEIGHT/2);
			graphics.moveTo(WIDTH/2, 0);
			graphics.lineTo(WIDTH/2, HEIGHT);
			
			// draw datapoints. Draw support vectors larger
			for (var i:int = 0; i < N; i++) 
			{
			  if (labels[i] == 1)
			  {
				  graphics.beginFill(0xB5EE4A);
			  }
			  else
			  {
				  graphics.beginFill(0x496AEF);
			  }
			  
			  if (svm.alpha[i] > 1e-2) 
			  {
				  graphics.lineStyle(3, 0);; // distinguish support vectors
			  }
			  else 
			  {
				  graphics.lineStyle(1, 0);
			  }
			  
			  graphics.drawCircle(data[i][0]*ss+WIDTH/2, data[i][1]*ss+HEIGHT/2, Math.floor(3+svm.alpha[i]*5.0/svmC));
			}
			
			// if linear kernel, draw decision boundary and margin lines
			if (kernelid == 0) 
			{
			  var xs:Array = [-5, 5];
			  var ys:Array = [0, 0];
			  ys[0] = ( -wb.b - wb.w[0] * xs[0]) / wb.w[1];
			  ys[1] = ( -wb.b - wb.w[0] * xs[1]) / wb.w[1];
			  graphics.beginFill(0);
			  graphics.lineStyle(1, 0);
			  // wx+b=0 line
			  graphics.moveTo(xs[0] * ss + WIDTH / 2, ys[0] * ss + HEIGHT / 2);
			  graphics.lineTo(xs[1] * ss + WIDTH / 2, ys[1] * ss + HEIGHT / 2);
			  // wx+b=1 line
			  graphics.moveTo(xs[0] * ss + WIDTH / 2, (ys[0] - 1.0 / wb.w[1]) * ss + HEIGHT / 2);
			  graphics.lineTo(xs[1] * ss + WIDTH / 2, (ys[1] - 1.0 / wb.w[1]) * ss + HEIGHT / 2);
			  // wx+b=-1 line
			  graphics.moveTo(xs[0] * ss + WIDTH / 2, (ys[0] + 1.0 / wb.w[1]) * ss + HEIGHT / 2);
			  graphics.lineTo(xs[1] * ss + WIDTH / 2, (ys[1] + 1.0 / wb.w[1]) * ss + HEIGHT / 2);
			  
			  // draw margin lines for support vectors. The sum of the lengths of these
			  // lines, scaled by C is essentially the total hinge loss.
			  for (i = 0; i < N; i++) 
			  {
				if (svm.alpha[i] < 1e-2)
				{
					continue;
				}
				if (labels[i] == 1) 
				{
				  ys[0] = (1 -wb.b - wb.w[0] * xs[0]) / wb.w[1];
				  ys[1] = (1 -wb.b - wb.w[0] * xs[1]) / wb.w[1];
				} 
				else 
				{
				  ys[0] = ( -1 -wb.b - wb.w[0] * xs[0]) / wb.w[1];
				  ys[1] = ( -1 -wb.b - wb.w[0] * xs[1]) / wb.w[1];
				}
				var u:Number = (data[i][0] - xs[0]) * (xs[1] - xs[0]) + (data[i][1] - ys[0]) * (ys[1] - ys[0]);
				u = u / ((xs[0] - xs[1]) * (xs[0] - xs[1]) + (ys[0] - ys[1]) * (ys[0] - ys[1]));
				var xi:Number = xs[0] + u * (xs[1] - xs[0]);
				var yi:Number = ys[0] + u * (ys[1] - ys[0]);
				graphics.moveTo(data[i][0] * ss + WIDTH / 2, data[i][1] * ss + HEIGHT / 2);
				graphics.lineTo(xi * ss + WIDTH / 2, yi * ss + HEIGHT / 2);
			  }
			  
			}
			
		}
		
		
	}
	
}