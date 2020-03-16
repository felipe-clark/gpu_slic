#include "../include/slic.h"

void enforce_label_connectivity(own_data* o_own_data, const int width,
    const int height, own_data* n_own_data, int n_spx)
{
	const int dx4[4] = {-1,  0,  1,  0};
	const int dy4[4] = { 0, -1,  0,  1};

	const int sz = width*height;
	const int SUPSZ = sz/n_spx;
	
	int label(0);
	int* xvec = new int[sz];
	int* yvec = new int[sz];
	int oindex(0);
	int adjlabel(0);//adjacent label
	for( int j = 0; j < height; j++ )
	{
		for( int k = 0; k < width; k++ )
		{
			if(!n_own_data[oindex].isValid())
			{
                                n_own_data[oindex].setLabel(label);
				//--------------------
				// Start a new segment
				//--------------------
				xvec[0] = k;
				yvec[0] = j;
				//-------------------------------------------------------
				// Quickly find an adjacent label for use later if needed
				//-------------------------------------------------------
				{for( int n = 0; n < 4; n++ )
				{
					int x = xvec[0] + dx4[n];
					int y = yvec[0] + dy4[n];
					if( (x >= 0 && x < width) && (y >= 0 && y < height) )
					{
						int nindex = y*width + x;
						if(n_own_data[nindex].isValid()) adjlabel = n_own_data[nindex].getLabel();
					}
				}}

				int count(1);
				for( int c = 0; c < count; c++ )
				{
					for( int n = 0; n < 4; n++ )
					{
						int x = xvec[c] + dx4[n];
						int y = yvec[c] + dy4[n];

						if( (x >= 0 && x < width) && (y >= 0 && y < height) )
						{
							int nindex = y*width + x;

							if( !n_own_data[nindex].isValid() && o_own_data[oindex] == o_own_data[nindex] )
							{
								xvec[count] = x;
								yvec[count] = y;
								n_own_data[nindex].setLabel(label);
								count++;
							}
						}

					}
				}
				//-------------------------------------------------------
				// If segment size is less then a limit, assign an
				// adjacent label found before, and decrement label count.
				//-------------------------------------------------------
				if((label == n_spx) || (count <= SUPSZ >> 2))
				{
					for( int c = 0; c < count; c++ )
					{
						int ind = yvec[c]*width+xvec[c];
						n_own_data[ind].setLabel(adjlabel);
					}
					label--;
				}
				label++;
			}
			oindex++;
		}
	}

	if(xvec) delete [] xvec;
	if(yvec) delete [] yvec;
}
