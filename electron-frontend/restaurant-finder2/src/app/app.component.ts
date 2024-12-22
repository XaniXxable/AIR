import { Component, CUSTOM_ELEMENTS_SCHEMA } from '@angular/core';

import { MatInputModule } from '@angular/material/input';
import { CommonModule } from '@angular/common';
import { RestaurantComponent } from './restaurant/restaurant.component';
import { RestaurantMetaData } from '../interfaces/RestaurantMetaData';
import { NgxSpinnerModule } from 'ngx-spinner';
import { MatMenuModule } from '@angular/material/menu';
import { MatIconModule } from '@angular/material/icon';
import { MatCheckboxModule } from '@angular/material/checkbox';

import { NgxSpinnerService } from 'ngx-spinner';
import { FormControl, FormGroup, FormsModule, ReactiveFormsModule } from '@angular/forms';
import { Filter, RestaurantFinderRequest } from '../interfaces/RestaurantFinderRequest';

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
    ReactiveFormsModule
  ],
  schemas: [CUSTOM_ELEMENTS_SCHEMA],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css'
})

export class AppComponent {

  public query: string = "";
  public latestQuery: string = "";
  public queryResults: RestaurantMetaData[] = []; 
  
  private additionalOptionsForm: FormGroup = new FormGroup({});
  private additionalOptions: any[] = [
    {name: 'Family-Friendly', selected: false},
    {name: 'Pet-Friendly', selected: false}, 
    {name: 'Cow-Friendly', selected: false},
    {name: 'Cat-Friendly', selected: false},
  ]

  constructor (private spinner: NgxSpinnerService) {
    this.additionalOptions.map((item) => {
      this.additionalOptionsForm.addControl(item.name, new FormControl(item.selected, {nonNullable: true}));
    });
  }

  get checkableOptions() {
    return Object.keys(this.additionalOptionsForm.controls);
  }

  public reset() {
    this.query = "";
    this.queryResults = [];

    this.toggleOffCheckboxes();
  }

  public getFormControl(key: string) {
    return this.additionalOptionsForm.controls[key] as FormControl;
  }

  public toggleOffCheckboxes() {
    this.additionalOptionsForm.reset();
  }

  delay(ms: number) {
    return new Promise( resolve => setTimeout(resolve, ms) );
  }

  public async onEnter() {
    if (this.query === "")
      return;

    console.log(this.query);
    
    this.spinner.show();

    var requestBody: RestaurantFinderRequest = {
      UserInput: this.query,
      Filters: this.evaluateAdditionalOptionsForm()
    } 
    
    console.log(requestBody);

    // BE call
    await this.delay(2000);

    this.spinner.hide();

    this.latestQuery = this.query;
    this.queryResults = [
      {
        Location: "Graz",
        Name: "Tree Amici",
        Image: "../assets/images/restaurant-types/italian.jpg",
        Reviews: 400,
        Type: "Italian"
      },
      {
        Location: "Vienna",
        Name: "McDonalds",
        Image: "../assets/images/restaurant-types/american.jpg",
        Reviews: 1001,
        Type: "American"
      },
      {
        Location: "Graz",
        Name: "Meet",
        Image: "../assets/images/restaurant-types/asian.jpg",
        Reviews: 30,
        Type: "Asian"
      },
      {
        Location: "Graz",
        Name: "Don Camillo",
        Image: "../assets/images/restaurant-types/italian.jpg",
        Reviews: 1050,
        Type: "Italian"
      }
    ]
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
