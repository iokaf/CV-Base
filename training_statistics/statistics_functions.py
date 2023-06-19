"""
    This module is for cuts data analysis.

"""


from typing import List, Dict
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class StatisticsFunctions:
    """This class is responsible for analyzing cuts in examinations and returning
    the data in a usable format.
    """
    def __init__(self, annotations: List[Dict]):
        """Arguments:
        ----------
        annotations (List[Dict]):
            A list of dictionaries, each of which represents a snare cut.
        """

        self.__annotations = annotations
        self.all_filenames = self.get_all_filename()
        self.max_id = self.max_polyp_id()
        self.all_cuts = self.number_of_cuts(data = self.all_filenames)
        self.distribution_number_cuts = self.distribution_number_of_cuts(self.all_cuts)

    def calculate_number_of_cuts(self) -> int:
        """Calculates the total number of cuts of all examinations from the list of annotations.

        Returns:
        --------
        int: The number of cuts in the file.
        """

        return len(self.__annotations)

    def number_of_first_cut(self) -> int: #(annotations: List[Dict]) -> int:
        """Calculate the number of first cuts

        Returns:
        ------------
            int: The number of first cuts in the file.
        """

        return sum(1 for annotation in self.__annotations if annotation["first_cut"])

    def max_polyp_id(self) -> int:
        """Get maximum polyp ID(number) in examinations.

        Returns:
        ------------
            int: A max polyp ID.
        """

        polypid = [annotation["polyp_id"] for annotation in self.__annotations]

        return max(polypid)

    def get_all_filename(self) -> List[str]:
        """Get filenames (not repetitive).

        Returns:
        ------------
            list: A list of unique filename.
        """

        filenameList = np.unique([annotation["video_filename"] for annotation in self.__annotations])

        return filenameList


    def number_of_cuts(self, data,tool="all") -> list:
        """Get number of cuts of each examination/video.

        Returns:
        ------------
            list: A list of number of cuts of each examination.
        """
        #get_all_filename = get_all_filename(self.__annotations)
        #max_polyp_id = max_polyp_id(self.__annotations)

        number_of_cuts = []

        #for name in self.all_filenames:
        for name in data:
            # ReDo?
            for i in range(self.max_id+1):
                if tool=="all":
                    the_cuts = [cut for cut in self.__annotations if cut["video_filename"] == name and cut["polyp_id"] == i]
                elif tool == "cold snare":
                    the_cuts = [cut for cut in self.__annotations if cut["video_filename"] == name and cut["polyp_id"] == i and cut['tool'] == 'cold snare']
                elif tool == "hot snare":
                    the_cuts = [cut for cut in self.__annotations if cut["video_filename"] == name and cut["polyp_id"] == i and cut['tool'] == 'hot snare']
                else:
                    print("Choose a tool.")

                if len(the_cuts) != 0:
                    number_of_cuts.append(len(the_cuts))

        return number_of_cuts

    def distribution_number_of_cuts(self,list_all_cuts) -> list:
        """Get number of polyps has 1,2,3... cuts (distribution). This function is mainly to check the plot.

        Arguments:
        ----------
        annotations (List[Dict]):
            A list of dictionaries, each of which represents a snare cut.

        Returns:
        ------------
            list: A list of number of polyps with 1,2,3,.. cuts.
        """

        distribution_number_of_cuts = []
        for i in range(max(list_all_cuts)+1):
            distribution_number_of_cuts.append(list_all_cuts.count(i))
        return distribution_number_of_cuts

    def get_filename_by_center(self, Center) -> list:
        """Retrieve filenames based on the specified center.
        
        Arguments:
        Center (str):
            The name of the center.
        
        Returns:
        list:
            A list of filenames corresponding to the specified center.
        """
        

        Boeck_Center = []
        Passek_Center = []
        Heil_Center =[]
        Ludwig_Center = []
        Stuttgart_Center = []
        Heubach_Center = []
        Simonis_Center = []
        Katharinen_Center =[]
        Other_Center = []

        for filename in self.all_filenames:
            if 'boeck' in filename.lower():
                Boeck_Center.append(filename)
            elif 'passek' in filename.lower():
                Passek_Center.append(filename)
            elif 'heil' in filename.lower():
                Heil_Center.append(filename)
            elif 'ludwig' in filename.lower():
                Ludwig_Center.append(filename)
            elif 'stuttgart' in filename.lower():
                Stuttgart_Center.append(filename)
            elif 'heubach' in filename.lower():
                Heubach_Center.append(filename)
            elif 'katharinen' in filename.lower():    
                Katharinen_Center.append(filename)
            elif 'simonis' in filename.lower():    
                Simonis_Center.append(filename) 
            else:
                Other_Center.append(filename)
                
        if Center.lower() == 'boeck':
            return Boeck_Center
        elif Center.lower() == 'passek':
            return Passek_Center      
        elif Center.lower() == 'heil':
            return Heil_Center
        elif Center.lower() == 'ludwig':
            return Ludwig_Center        
        elif Center.lower() == 'stuttgart':
            return Stuttgart_Center
        elif Center.lower() == 'heubach':
            return Heubach_Center
        elif Center.lower() == 'katharinen':
            return Katharinen_Center
        elif Center.lower() == 'simonis':
            return Simonis_Center
        else:
            return print(f"Please enter a valid center name. / check center not in the list:{Other_Center}")   
    
    
    
    
    
    def distribution_plot(self, cuts_data, title) -> None:
        """ plot the distribution of the number of cuts per polyp.
    
        Returns:
        ------------
            None
        """

        # make a histogram plot
        plot = sns.histplot(cuts_data, discrete= 1)
        plt.xlabel("Number of cuts")
        plt.ylabel("Count (Polyp)")
        plot.set_title(title)
        plt.xticks(range(min(cuts_data), max(cuts_data)+1))
        for p in plot.patches:
            # Get the height of the bar
            height = p.get_height()
            # Add the text annotation at the top of each bar
            if height:
                plt.annotate(str(height), (p.get_x() + p.get_width()/2, height), ha='center', va='bottom')
        plt.show()


    


