import { Component, Input } from '@angular/core';
import { RestaurantMetaData } from '../../interfaces/RestaurantMetaData';
import { CommonModule } from '@angular/common';
import { MatIconModule } from '@angular/material/icon'


@Component({
  selector: 'restaurant',
  standalone: true,
  imports: [
    CommonModule,
    MatIconModule
  ],
  templateUrl: './restaurant.component.html',
  styleUrl: './restaurant.component.css'
})
export class RestaurantComponent {

  @Input() restaurant: RestaurantMetaData | undefined;
}
