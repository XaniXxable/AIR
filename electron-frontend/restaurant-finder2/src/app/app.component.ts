import { Component, CUSTOM_ELEMENTS_SCHEMA, ViewChild } from '@angular/core';

import { RestaurantComponent } from './restaurant/restaurant.component';
import { RestaurantMetaData } from '../interfaces/RestaurantMetaData';

import { CommonModule } from '@angular/common';
import { MatInputModule } from '@angular/material/input';
import { NgxSpinnerModule } from 'ngx-spinner';
import { MatMenu, MatMenuModule, MatMenuTrigger } from '@angular/material/menu';
import { MatIconModule } from '@angular/material/icon';
import { MatCheckboxModule } from '@angular/material/checkbox';
import { MatSelectModule } from '@angular/material/select';

import { NgxSpinnerService } from 'ngx-spinner';
import { FormControl, FormGroup, FormsModule, ReactiveFormsModule } from '@angular/forms';
import { Filter, RestaurantFinderRequest } from '../interfaces/RestaurantFinderRequest';
import { HttpClientModule, HttpClient, HttpHeaders } from '@angular/common/http';
import { RestaurantFinderResponse } from '../interfaces/RestaurantFinderResponse';


@Component({
  selector: 'app-root',
  standalone: true,
  imports: [
    MatInputModule,
    CommonModule,
    RestaurantComponent,
    NgxSpinnerModule,
    FormsModule,
    MatMenuModule, 
    MatIconModule,
    MatCheckboxModule,
    MatSelectModule,
    ReactiveFormsModule,
    HttpClientModule
  ],
  schemas: [CUSTOM_ELEMENTS_SCHEMA],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css'
})

export class AppComponent {
  public query: string = "";
  public latestQuery: string = "";
  public queryResults: RestaurantMetaData[] = []; 
  
  public selectedSortingOption: string = "option1";

  private additionalOptionsForm: FormGroup = new FormGroup({});
  private additionalOptions: any[] = [
    {name: 'Family-Friendly', selected: false},
    {name: 'Pet-Friendly', selected: false}
  ]

  constructor (private spinner: NgxSpinnerService, private http: HttpClient) {
    this.additionalOptions.map((item) => {
      this.additionalOptionsForm.addControl(item.name, new FormControl(item.selected, {nonNullable: true}));
    });
  }

  get checkableOptions() {
    return Object.keys(this.additionalOptionsForm.controls);
  }

  get sortingOptions() {
    let options: string[] = [];
    
    Object.keys(this.queryResults[0]).forEach((key) => {
      if (key != "Image")
        options.push(key);
    })

    return options;
  }

  public reset() {
    this.query = "";
    this.latestQuery = "";
    this.queryResults = [];
    this.selectedSortingOption = "option1";

    this.toggleOffCheckboxes();
  }

  public sortResults(key: string): void {
    this.spinner.show();
    this.queryResults = this.insertionSort(key);
    this.spinner.hide();
  }

  // Src: https://dev.to/bugudiramu/a-developers-guide-to-sorting-algorithms-2kl9
  private insertionSort(key: string): RestaurantMetaData[] {
    let sortedQueryResults: RestaurantMetaData[] = this.queryResults.slice();
    const length: number = sortedQueryResults.length;

    for (let idx: number = 1; idx < length; idx++) {
      const currentElement: any = sortedQueryResults[idx];

      let col: number = idx - 1;
  
      while (col >= 0 && (sortedQueryResults[col] as any)[key] > currentElement[key]) {
        sortedQueryResults[col + 1] = sortedQueryResults[col];
        col--;
      }
  
      sortedQueryResults[col + 1] = currentElement;
    }

    if (this.selectedSortingOption === "option2")
      sortedQueryResults = sortedQueryResults.reverse()
    
    return sortedQueryResults;
  }

  public getFormControl(key: string) {
    return this.additionalOptionsForm.controls[key] as FormControl;
  }

  public toggleOffCheckboxes() {
    this.additionalOptionsForm.reset();
  }

  // delay(ms: number) {
  //   return new Promise( resolve => setTimeout(resolve, ms) );
  // }

  public async onEnter() {
    if (this.query === "")
      return;

    this.spinner.show();

    var requestBody: string = JSON.stringify({
      UserInput: this.query,
      Filters: this.evaluateAdditionalOptionsForm()
    }) 
    
    console.log(requestBody);

    // BE call
    this.http.post("http://127.0.0.1:5000/restauratnts/", requestBody, 
      {
        headers: new HttpHeaders({
          'Content-Type':  'application/json'
        })
      })
    .subscribe((res: any) => {
        console.log(res);
        
        this.queryResults = res["Data"];
        this.latestQuery = this.query;
        this.spinner.hide();
    });


    // this.queryResults = [
    //   {
    //     Location: "Graz",
    //     Name: "Tre Amici",
    //     Image: "assets/images/restaurant-types/italian.jpg",
    //     Reviews: 400,
    //     Type: "Italian"
    //   },
    //   {
    //     Location: "Vienna",
    //     Name: "McDonalds",
    //     Image: "assets/images/restaurant-types/default.jpg",
    //     Reviews: 1001,
    //     Type: "American"
    //   },
    //   {
    //     Location: "Graz",
    //     Name: "Meet",
    //     Image: "assets/images/restaurant-types/asian.jpg",
    //     Reviews: 30,
    //     Type: "Asian"
    //   },
    //   {
    //     Location: "Graz",
    //     Name: "Don Camillo",
    //     Image: "assets/images/restaurant-types/italian.jpg",
    //     Reviews: 1050,
    //     Type: "Italian"
    //   }
    // ]
  }

  private evaluateAdditionalOptionsForm(): Filter[] {
    var filters: Filter[] = [];
    this.checkableOptions.forEach(key => {
      var filter = this.additionalOptionsForm.controls[key];
      filters.push({
        FilterName: key,
        FilterValue: filter.value
      })
    });
    
    return filters;
  }

}
